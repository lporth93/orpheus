import numpy as np
from itertools import combinations_with_replacement
import os
import site

def convertunits(unit_in, unit_target):
    '''unit can be '''
    vals = {'rad': 180./np.pi, 
            'deg': 1.,
            'arcmin': 1./60.,
            'arcsec': 1./60./60.}
    assert((unit_in in vals.keys()) and (unit_target in vals.keys()))
    return vals[unit_in]/vals[unit_target]

def flatlist(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatlist(i))
        else: rt.append(i)
    return rt

def get_site_packages_dir():
        return [p for p  in site.getsitepackages()
                if p.endswith(("site-packages", "dist-packages"))][0]

def search_file_in_site_package(directory, package):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(package):
                return os.path.join(root, file)
    return None

def gen_thetacombis_fourthorder(nbinsr, nthreads, batchsize, batchsize_max, ordered=True, custom=None, verbose=False):
        
        # Allocate selector for custom bins
        if custom is None:
            customsel = np.ones(nbinsr*nbinsr*nbinsr, dtype=bool)
        else:
            custom = custom.astype(int)
            assert(np.max(custom)<nbinsr*nbinsr*nbinsr)
            assert(np.min(custom)>=0)
            customsel = np.zeros(nbinsr*nbinsr*nbinsr, dtype=bool)
            customsel[custom] = True
            
        # Build the bins    
        allelbs = []
        thetacombis_batches = []
        nbinsr3 = 0
        cutlo_2 = 0
        cutlo_3 = 0
        tmpind = 0
        for elb1 in range(nbinsr):
            for elb2 in range(nbinsr):
                for elb3 in range(nbinsr):
                    valid = True
                    if ordered:
                        if elb1>elb2 or elb1>elb3 or elb2>elb3:
                            valid = False
                    if valid and customsel[tmpind]:
                        thetacombis_batches.append([tmpind])
                        allelbs.append([elb1,elb2,elb3])
                        nbinsr3 += 1
                    tmpind += 1
        thetacombis_batches = np.asarray(thetacombis_batches,dtype=np.int32)
        allelbs = np.asarray(allelbs,dtype=np.int32)
        if batchsize is None:
            batchsize = min(nbinsr3,min(batchsize_max,nbinsr3/nthreads))
            if verbose:
                print("Using batchsize of %i for radial bins"%batchsize)
        if batchsize==batchsize_max:
            nbatches = np.int32(np.ceil(nbinsr3/batchsize))
        else:
            nbatches = np.int32(nbinsr3/batchsize)
        #thetacombis_batches = np.arange(nbinsr3).astype(np.int32)
        cumnthetacombis_batches = (np.arange(nbatches+1)*nbinsr3/(nbatches)).astype(np.int32)
        nthetacombis_batches = (cumnthetacombis_batches[1:]-cumnthetacombis_batches[:-1]).astype(np.int32)
        cumnthetacombis_batches[-1] = nbinsr3
        nthetacombis_batches[-1] = nbinsr3-cumnthetacombis_batches[-2]
        thetacombis_batches = thetacombis_batches.flatten().astype(np.int32)
        nbatches = len(nthetacombis_batches)
        
        return nbinsr3, allelbs, thetacombis_batches, cumnthetacombis_batches, nthetacombis_batches, nbatches

def gen_n2n3indices_Upsfourth(nmax):
        """ List of flattened indices corresponding to selection """
        nmax_alloc = 2*nmax+1
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
        _n2s = np.argwhere(reconstructed>0)[:,0].astype(np.int32)-nmax_alloc
        _n3s = np.argwhere(reconstructed>0)[:,1].astype(np.int32)-nmax_alloc
        return _shape, _inds, _n2s, _n3s

def symmetrize_map3_multiscale(map3, return_list=False):
    """
    Symmetrizes third-order aperture mass over tomographic bin combinations
    and radial bin combinations
    Assumes map3 to be of shape (8, nbinsz**3, nbinsr**3)
    """
    
    nbinsz = int(round((map3.shape[1])**(1/3)))
    nbinsr = int(round((map3.shape[2])**(1/3)))

    # Get unique combinations of indices (e.g., r1 <= r2 <= r3)
    r_combs = np.array(list(combinations_with_replacement(range(nbinsr), 3)))
    z_combs = np.array(list(combinations_with_replacement(range(nbinsz), 3)))
    indr3, indz3 = r_combs.shape[0], z_combs.shape[0]

    # Get permutations for each unique combination
    perm_map = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1],
                         [1, 0, 2], [2, 1, 0], [0, 2, 1]])
    
    # Create arrays of permuted 3D indices
    # Shape is (6, n_combinations, 3)
    r_perms_3d = r_combs[:, perm_map].transpose(1, 0, 2)
    z_perms_3d = z_combs[:, perm_map].transpose(1, 0, 2)

    # Convert 3D permuted indices to flat indices
    r_powers = np.array([nbinsr**2, nbinsr, 1])
    sel_foots = np.dot(r_perms_3d, r_powers) # Shape: (6, indr3)
    z_powers = np.array([nbinsz**2, nbinsz, 1])
    zcombis = np.dot(z_perms_3d, z_powers) # Shape: (6, indz3)

    # Do the averaging
    # Shape is (8, 6_z_perms, indz3, 6_r_perms, indr3)
    all_perms_data = map3[:, zcombis][:, :, :, sel_foots]
    map3_symm = np.mean(all_perms_data, axis=(1, 3))

    # Allocate final result
    res = (map3_symm,)
    if return_list:
        # Rearrange and split the data to match original list format
        list_data = all_perms_data.transpose(3, 0, 1, 4, 2)
        map3_list = [arr.squeeze(axis=-1) for arr in np.split(list_data, indz3, axis=-1)]
        res += (map3_list,)
        
    return res