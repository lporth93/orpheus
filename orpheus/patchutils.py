# Here we collect some utils for mapping the a full-sky survey to a set of overlapping patches
# In the middle term much of this functionality should be included in the orpheus code

import os
import sys
from multiprocessing.shared_memory import SharedMemory
from astropy.coordinates import SkyCoord
from healpy import ang2pix, pix2vec, nside2pixarea, nside2resol, query_disc, Rotator, nside2npix
import numpy as np
from pathlib import Path
import pickle
from time import time
import warnings as _warnings
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed
import joblib as _joblib

from sklearn.cluster import KMeans

# Detect whether joblib supports streaming results (added in 1.2).
# We use this for per-result progress output without holding all results in RAM.
_JOBLIB_HAS_GENERATOR = (tuple(int(x) for x in _joblib.__version__.split(".")[:2]) >= (1, 2))

def _shm_create(arr):
    """
    Copy a numpy array into a new POSIX SharedMemory block.

    Returns
    -------
    shm : SharedMemory
        Keep alive in the parent until all workers have finished, then call
        shm.close() and shm.unlink().
    spec : tuple  (name, shape, dtype_str)
        Everything needed to re-attach in a worker process.
    """
    shm  = SharedMemory(create=True, size=max(arr.nbytes, 1))
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
    return shm, (shm.name, arr.shape, arr.dtype.str)

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
                        kmeanshp_maxiter=1000, kmeanshp_tol=1e-10, kmeanshp_randomstate=42, healpix_nside=8, n_workers=16):
    """
    Decomposes a spherical catalog in ~equal-area patches with a buffer region.

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
    method: {'kmeans_healpix', 'kmeans_treecorr', 'healpix'}
        Patch-assignment algorithm.
    kmeanshp_maxiter: int
        KMeans maximum iterations (kmeans_healpix method only).
    kmeanshp_tol: float
        KMeans convergence tolerance (kmeans_healpix method only).
    kmeanshp_randomstate: int
        KMeans random seed (kmeans_healpix method only).
    healpix_nside: int
        Healpix nside for patch assignment (healpix method only).
    n_workers: int or None
        Number of parallel worker processes for buffer construction.
        Follows joblib convention: None/-1 --> all CPUs, 1-->sequential.
        Defaults to 16.

    Returns
    -------
    cat_patchindices: dict
        A dictionary containing information about the individual patches,
        as well as the galaxy indices that are assigned to the inner region
        and to the buffer region of each individual patch

    Notes
    -----
    * Uses joblib with the loky backend, which is safe with JAX and any other multithreaded library
    * Large read-only arrays (theta, phi, ...) are placed in POSIX SharedMemory segments so
      physically the same RAM pages are mapped into all worker proecsses with zero copies.
    * Using too many workers can cause significant time spent to start the workers, which is done
      sequentially by loky. 
    * Choosing a small value of nside_hash will result in a larger extension of 
      the patches than necessary while choosing a large value increases the 
      runtime. A good compromise is to choose nside_hash such that its resolution 
      is a few times smaller than the buffer region of the patches    
    """


    ## Step 1: Define theta/phi used lateron. ##
    if verbose:
        print("Computing sky coordinates")
        t1 = time()
    eq    = SkyCoord(ra_deg, dec_deg, frame='galactic', unit='deg')
    l, b  = eq.galactic.l.value, eq.galactic.b.value
    theta = np.radians(90. - b).astype(np.float64)
    phi   = np.radians(l).astype(np.float64)
    if verbose:
        print("Took %.3f seconds" % (time() - t1))

    ## Step2: Define inner region of patches ##
    # Run treecorrs k-means implementation
    if method == 'kmeans_treecorr':
        try:
            import treecorr
            if verbose:
                print("Computing patch assignment via treecorr KMeans")
                t1 = time()
            cat       = treecorr.Catalog(
                ra=ra_deg, dec=dec_deg,
                ra_units='deg', dec_units='deg',
                npatch=npatches,
            )
            patchinds = cat.patch
            if verbose:
                print("Took %.3f seconds" % (time() - t1))
        except ImportError:
            print("treecorr not available; switching to kmeans_healpix")
            method = 'kmeans_healpix'
    # Run standard k-means on catalog reduced to healpix pixels
    if method == 'kmeans_healpix':
        if verbose:
            print("Computing patch assignment via KMeans on HEALPix pixels")
            t1 = time()
        # Step A: Reduce discrete theta/phi to unique healpix pixels and transform those to to 3D positions
        nside_kmeans = 2048
        hpx_inds     = ang2pix(nside_kmeans, theta, phi)
        hpx_uinds    = np.unique(hpx_inds)
        X = np.array(pix2vec(nside=nside_kmeans, ipix=hpx_uinds, nest=False)).T
        # Step B: Run standard kmeans algorithm on the healpix pixels
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
        clust = KMeans(
            n_clusters=npatches,
            init='k-means++',
            n_init='auto',
            max_iter=kmeanshp_maxiter,
            tol=kmeanshp_tol,
            verbose=0,
            random_state=kmeanshp_randomstate,
            copy_x=True,
            algorithm='lloyd')
        # Temorarily limit max number of OMP here as KMeans per default chooses all available
        # cores and might crash in case scipy has not been compiled to handle this.
        # Also I observed that KMeans becomes fairly inefficient for this many cores anyways.
        with threadpool_limits(limits=32, user_api='openmp'):
            clustinds = clust.fit_predict(X, y=None, sample_weight=None)
        # Step C: Map the pixel centers back to the galaxy indices
        hashmap   = np.vectorize({u: c for u, c in zip(hpx_uinds, clustinds)}.get)
        patchinds = hashmap(hpx_inds)
        if verbose:
            print("Took %.3f seconds"%(time()-t1))
    # Simply assign to healpix pixel. Fast and stable, but patchareas might strongly vary in size.
    elif method == 'healpix':
        if verbose:
            print("Computing patch assignment via HEALPix pixel assignment")
            t1 = time()
        # Filter out empty patches -- this happens if the catalog does not cover the full sphere.
        _, patchinds = np.unique(ang2pix(healpix_nside, theta, phi), return_inverse=True)
        npatches = int(patchinds.max()) + 1
        if verbose:
            print("Took %.3f seconds" % (time() - t1))
    else:
        raise NotImplementedError(f"Unknown method: {method!r}")

    ## Step 3: Retrieve patch centers ##
    if verbose:
        print("Computing patch centres")
        t1 = time()
    if method == 'kmeans_treecorr':
        _patchcenters = cat.patch_centers
    else:
        counts = np.bincount(patchinds, minlength=npatches).astype(float)
        ra_sum = np.bincount(patchinds, weights=ra_deg,  minlength=npatches)
        dec_sum = np.bincount(patchinds, weights=dec_deg, minlength=npatches)
        ra_mean = np.divide(ra_sum,  counts, out=np.full(npatches, np.nan), where=counts > 0)
        dec_mean = np.divide(dec_sum, counts, out=np.full(npatches, np.nan), where=counts > 0)
        _patchcenters = np.column_stack((ra_mean, dec_mean))
    if verbose:
        print("Took %.3f seconds"%(time()-t1))

    ## Step 4: Map catalog to healpix grid ##
    if verbose:
        print("Mapping catalogue to HEALPix hash grid (nside=%d)" % nside_hash)
        t1 = time()
    cat_indices = ang2pix(nside_hash, theta, phi)
    _pixarea    = nside2pixarea(nside_hash, degrees=True)
    _pixreso    = nside2resol(nside_hash, arcmin=True)
    ext_buffer  = (patchextend_arcmin + _pixreso) * np.pi / 180.0 / 60.0
    if verbose:
        print("Took %.3f seconds" % (time() - t1))

    ## Step 5: Build a hash connecting the galaxies residing in each healpix pixel ## 
    # In order to allow for parallelisation we store the hash in three arrays, similar
    # to what we do in the C code.
    if verbose:
        print("Building index hash")
        t1 = time()
    sort_hash = np.argsort(cat_indices, kind='stable')
    csr_unique_pix, _, pix_counts = np.unique(cat_indices[sort_hash], return_index=True, return_counts=True)
    csr_offsets = np.zeros(len(csr_unique_pix) + 1, dtype=np.int64)
    np.cumsum(pix_counts, out=csr_offsets[1:])
    csr_galinds = sort_hash.astype(np.int64)
    if verbose:
        print("Took %.3f seconds" % (time() - t1))

    ## Step 6: Build a similar hash for the patchindices of the gals ##
    # This will be used to quickly distribute the galaxies acreoss the workers
    # without them having to repeatadly read from the same array.
    if verbose:
        print("Pre-splitting inner galaxy indices")
        t1 = time()
    sort_patch  = np.argsort(patchinds, kind='stable').astype(np.int64)
    _, _, patch_counts = np.unique(patchinds[sort_patch], return_index=True, return_counts=True)
    galinds_inner_offsets = np.zeros(npatches + 1, dtype=np.int64)
    np.cumsum(patch_counts, out=galinds_inner_offsets[1:])
    galinds_inner_flat = sort_patch   # This will be used by workers that slice by offsets
    if verbose:
        print("Took %.3f seconds" % (time() - t1))

    ## Step 7: Put large read-obly arrays in shared memory ##
    # Layout as such because workers attach to the same physical RAM pages via named segments.
    if verbose:
        print("Creating shared memory segments")
        t1 = time()
    shm_objects = {}
    shm_specs   = {}
    for key, arr in {
        'theta': theta,
        'phi': phi,
        'csr_unique_pix': csr_unique_pix.astype(np.int64),
        'csr_offsets': csr_offsets,
        'csr_galinds': csr_galinds,
        'galinds_inner_flat': galinds_inner_flat}.items():
        shm, spec  = _shm_create(arr)
        shm_objects[key] = shm
        shm_specs[key] = spec
    if verbose:
        print("Took %.3f seconds" % (time() - t1))

    ## Step 7: Define structure for results ##
    cat_patchindices = {
        "info": {
            "patchextend_deg": patchextend_arcmin / 60.0,
            "nside_hash": nside_hash,
            "method" : method,
            "kmeanshp_maxiter": kmeanshp_maxiter,
            "kmeanshp_tol": kmeanshp_tol,
            "kmeanshp_randomstate": kmeanshp_randomstate,
            "healpix_nside": healpix_nside,
            "patchcenters": _patchcenters,
            "patchareas": np.zeros(npatches, dtype=float),
            "patch_ngalsinner": np.zeros(npatches, dtype=int),
            "patch_ngalsouter": np.zeros(npatches, dtype=int)},
        "patches": {p: {} for p in range(npatches)}}

    ## Step 8: Create buffer in parallel ##
    # joblib/loky: cloudpickle serialisation, clean worker processes,
    # safe with JAX and any multithreaded library.
    if verbose:
        print("Building buffer around patches")
        t1 = time()

    # Use all availabel CPUs if nothing else specified (following joblib convention)
    n_jobs = -1 if n_workers is None else n_workers

    # Bind fixed arguments so each task only carries the patch index
    def _task(elpatch):
        return delayed(_process_patch)(
            elpatch, shm_specs, galinds_inner_offsets,
            nside_hash, ext_buffer, _pixarea, )

    def _collect(elpatch, n_inner, galinds_outer, patch_area):
        s = int(galinds_inner_offsets[elpatch])
        e = int(galinds_inner_offsets[elpatch + 1])
        cat_patchindices["patches"][elpatch]["inner"] = galinds_inner_flat[s:e].copy()
        cat_patchindices["patches"][elpatch]["outer"] = galinds_outer
        cat_patchindices["info"]["patchareas"][elpatch] = patch_area
        cat_patchindices["info"]["patch_ngalsinner"][elpatch] = n_inner
        cat_patchindices["info"]["patch_ngalsouter"][elpatch] = len(galinds_outer)

    try:
        _warnings.filterwarnings(
            'ignore',
            message=r'os\.fork\(\) was called',
            category=RuntimeWarning,
            module=r'loky',
        )
        if _JOBLIB_HAS_GENERATOR:
            # joblib version >= 1.2 --> consume results as they arrive (streaming).
            # In this case the peak memory is ~ n_workers in-flight, not npatches.
            gen = Parallel(n_jobs=n_jobs, backend='loky',return_as='generator_unordered')(_task(i) for i in range(npatches))
            for i, result in enumerate(gen):
                if verbose:
                    sys.stdout.write("\r%i/%i"%(i+1, npatches))
                    sys.stdout.flush()
                _collect(*result)
        else:
            # joblib version < 1.2: collect all results, then process.
            results = Parallel(n_jobs=n_jobs, backend='loky')(_task(i) for i in range(npatches))
            for i, result in enumerate(results):
                if verbose:
                    sys.stdout.write("\r%i/%i" % (i + 1, npatches))
                    sys.stdout.flush()
                _collect(*result)

    # Release shared memory segments regardless of success or failure.
    # close() --> unmap from parent's address space.
    # unlink() --> delete the OS-level named object (free RAM).
    # Workers already closed their handles when _process_patch returned.
    finally:
        for shm in shm_objects.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
    if verbose:
        print("\nTook %.3f seconds" % (time() - t1))

    return cat_patchindices

def _process_patch(elpatch, shm_specs, galinds_inner_offsets,
                   nside_hash, ext_buffer, pixarea):
    """
    Compute galaxy indices of buffer region for one patch.

    Parameters
    ----------
    elpatch: int
    shm_specs: dict {key: (name, shape, dtype_str)}
        Specs for all large read-only arrays.
    galinds_inner_offsets : np.ndarray, shape (npatches+1,)
        Start/end offsets into galinds_inner_flat for each patch.
    nside_hash, ext_buffer, pixarea : scalars
    """
    shm_handles = []
    arrs        = {}
    try:
        # Attach to all SharedMemory segments
        for key, (name, shape, dtype) in shm_specs.items():
            shm = SharedMemory(name=name)
            shm_handles.append(shm)
            arrs[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # Get indices of gals within inner patch
        s = int(galinds_inner_offsets[elpatch])
        e = int(galinds_inner_offsets[elpatch + 1])
        galinds_inner = arrs['galinds_inner_flat'][s:e]
        n_inner = len(galinds_inner)

        if n_inner == 0:
            return elpatch, 0, np.empty(0, dtype=np.int64), 0.

        # Get healpix pixels covered by inner galaxies
        patch_hpx  = np.unique(
            ang2pix(nside_hash, arrs['theta'][galinds_inner], arrs['phi'][galinds_inner]))
        patch_area = float(pixarea*len(patch_hpx))

        # Find healpix pixels in extended patch
        ext_pixels: set = set()
        for pix in patch_hpx:
            ext_pixels.update(query_disc(nside=nside_hash, vec=pix2vec(nside_hash, pix), radius=ext_buffer))

        # Collect galaxies in extended pixels via CSR binary search
        unique_pix  = arrs['csr_unique_pix']
        csr_offsets = arrs['csr_offsets']
        csr_galinds = arrs['csr_galinds']
        parts = []
        for pix in ext_pixels:
            idx = int(np.searchsorted(unique_pix, pix))
            if idx < len(unique_pix) and unique_pix[idx] == pix:
                parts.append(csr_galinds[csr_offsets[idx]:csr_offsets[idx + 1]])

        if parts:
            # Make sure that galaxies already appearing as inner are excluded from outer
            gal_ext = np.unique(np.concatenate(parts))
            galinds_outer = gal_ext[~np.isin(gal_ext, galinds_inner, assume_unique=True)]
        else:
            galinds_outer = np.empty(0, dtype=np.int64)

        # galinds_inner is a view of SharedMemory; copy before the shm handle
        # is closed in the finally block so the caller receives owned data.
        return elpatch, n_inner, galinds_outer.copy(), patch_area

    # Unmap workers address space
    finally:
        for shm in shm_handles:
            shm.close()

def gen_cat_patchindices_old(ra_deg, dec_deg, npatches, patchextend_arcmin, nside_hash=128, verbose=False, method='kmeans_healpix',
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
    the patches than necessary while choosing a large value increases the 
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
        # Temorarily limit max number of OMP here as KMeans per default chooses all available
        # cores and might crash in case scipy has not been compiled to handle this.
        # Also I observed that KMeans becomes fairly inefficient for this many cores anyways.
        with threadpool_limits(limits=32, user_api="openmp"):   
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
        npatches = len(np.unique(patchinds).flatten())
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
        counts = np.bincount(patchinds, minlength=npatches).astype(float)
        ra_sum  = np.bincount(patchinds, weights=ra_deg,  minlength=npatches)
        dec_sum = np.bincount(patchinds, weights=dec_deg, minlength=npatches)
        ra_mean  = np.divide(ra_sum,  counts, out=np.full(npatches, np.nan), where=counts > 0)
        dec_mean = np.divide(dec_sum, counts, out=np.full(npatches, np.nan), where=counts > 0)
        _patchcenters = np.column_stack((ra_mean, dec_mean))
        #_patchcenters = np.array([[np.mean(ra_deg[ patchinds==patchind]), np.mean(dec_deg[ patchinds==patchind])] for patchind in range(npatches)])
    else:
        raise NotImplementedError
    
    cat_patchindices = {}
    cat_patchindices["info"] = {}
    cat_patchindices["info"]["patchextend_deg"] = patchextend_arcmin/60.
    cat_patchindices["info"]["nside_hash"] = nside_hash
    cat_patchindices["info"]["method"] = method
    cat_patchindices["info"]["kmeanshp_maxiter"] = kmeanshp_maxiter
    cat_patchindices["info"]["kmeanshp_tol"] = kmeanshp_tol
    cat_patchindices["info"]["kmeanshp_randomstate"] = kmeanshp_randomstate
    cat_patchindices["info"]["healpix_nside"] = healpix_nside
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

    # If method=="healpix" we might get empty patches. Here we filter those out
    if method=='healpix':
        print("Masking out empty patches")
        t1=time()
        sel_nonemptypatches = np.argwhere(cat_patchindices["info"]["patch_ngalsinner"]>0).flatten()
        n_nonemptypatches = len(sel_nonemptypatches)
        inds_nonemptypatches = {}
        for elpatch in range(n_nonemptypatches):
            inds_nonemptypatches[elpatch] = {}
            inds_nonemptypatches[elpatch]['inner'] = cat_patchindices["patches"][sel_nonemptypatches[elpatch]]["inner"][:]
            inds_nonemptypatches[elpatch]['outer'] = cat_patchindices["patches"][sel_nonemptypatches[elpatch]]["outer"][:]
        cat_patchindices["info"]["patchcenters"] = _patchcenters[sel_nonemptypatches]
        cat_patchindices["info"]["patchareas"] =  cat_patchindices["info"]["patchareas"][sel_nonemptypatches]
        cat_patchindices["info"]["patch_ngalsinner"] =cat_patchindices["info"]["patch_ngalsinner"][sel_nonemptypatches]
        cat_patchindices["info"]["patch_ngalsouter"] = cat_patchindices["info"]["patch_ngalsouter"][sel_nonemptypatches]
        cat_patchindices["patches"] = inds_nonemptypatches
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

    Returns
    -------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates.

    Notes
    -----
    This function is a generalised version of https://stackoverflow.com/a/50495134
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