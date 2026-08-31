# Here we collect all structures that are used to interface with the c code
# * To limit the number of different structures we set fields that are not
#   neeeded for a particular function to NULL in the python layer
# * To make sure that all arrays stay alive during the C call we also return 
#   the list of arrays that the caller must keep on the stack frame

import ctypes
import numpy as np

###########
# HELPERS #
###########
_p_f64 = ctypes.POINTER(ctypes.c_double)
_p_i32 = ctypes.POINTER(ctypes.c_int32)
_p_i64 = ctypes.POINTER(ctypes.c_int64)
_p_long = ctypes.POINTER(ctypes.c_long)
_p_c128 = ctypes.POINTER(ctypes.c_double)   # complex128 arrays passed as double*

# Just return a typed null pointer. This is safer than None for optional struct fields because
# ctypes somehow refuses to dereference it rather than silently coercing.
def _null(ptype):
    return ptype()

def _ptr_f64(arr): return arr.ctypes.data_as(_p_f64)
def _ptr_i32(arr): return arr.ctypes.data_as(_p_i32)
def _ptr_i64(arr): return arr.ctypes.data_as(_p_i64)
def _ptr_long(arr): return arr.ctypes.data_as(_p_long)


#######################
# STRUCTS DEFINITIONS # 
#######################
class MultiresoCatalog(ctypes.Structure):
    """Structure related to tracer catalogs and their reductions.
    """
    _fields_ = [
        # general content
        ("metric", ctypes.c_int32),
        ("nresos", ctypes.c_int32),
        ("ngal_resos", _p_i32),
        ("nbinsz", ctypes.c_int32),
        ("isinner_resos", _p_f64),
        ("weight_resos", _p_f64),
        ("zbin_resos", _p_i32),
        # Cartesian position coordinate (used for 2d and 3d flat methods)
        ("pos1_resos", _p_f64),
        ("pos2_resos",  _p_f64),
        ("pos3_resos", _p_f64),
        # 3D-vectors used for spherical ra/dec coordinates (used for sphierical geometries)
        ("vx_resos", _p_f64),
        ("vy_resos", _p_f64),
        ("vz_resos", _p_f64),
        # Helper quantities for the spin-2 geodesic bearing (used for sphierical geometries)
        ("ra_resos", _p_f64),
        ("sindec_resos", _p_f64),
        ("cosdec_resos", _p_f64),
        # Spin-2 tracer quantities (Used for SpinTracerCatalogs)
        ("e1_resos", _p_f64),
        ("e2_resos", _p_f64),
        ("weightsq_resos", _p_f64),
    ]


class NavHash(ctypes.Structure):
    """Quantities neccessary to navigate through a multihash nested spatial hashing structure
    """
    _fields_ = [
        ("metric", ctypes.c_int32),
        # Relevant for flat2d metric
        ("index_matcher", _p_i32),
        ("pixs_galind_bounds",  _p_i32),
        ("pix_gals", _p_i32),
        ("pix1_start", ctypes.c_double),
        ("pix1_d", ctypes.c_double),
        ("pix1_n", ctypes.c_int32),
        ("pix2_start", ctypes.c_double),
        ("pix2_d", ctypes.c_double),
        ("pix2_n", ctypes.c_int32),
        ("nregions", ctypes.c_int32),
        ("index_matcher_hash", _p_i32),
        ("filledregions", _p_i32),
        ("nfilledregions", ctypes.c_int32),
        # Additionally relevant for 3dbox ensemble of multihashes
        ("slab_offsets", _p_i32),
        ("rshift_bounds", _p_i32),
        ("nslabs", ctypes.c_int32),
        ("z0", ctypes.c_double),
        ("dpix_z", ctypes.c_double),
        # Relevant for spherical metric
        ("ncells_resos", _p_i32),
        ("nside_nav", _p_long),
        ("cell_pix", _p_long),
        ("cell_redbounds", _p_i32),
        ("rshift_red", _p_i32),
        ("rshift_cellpix", _p_i32),
        ("rshift_cellbounds", _p_i32),
    ]


class TreeResoParams(ctypes.Structure):
    """Quantities neccessary to describe the resolution levels of the multihashes
    """
    _fields_ = [
        ("nresos", ctypes.c_int32),
        ("nresos_grid", ctypes.c_int32),
        ("dpix1_resos", _p_f64),
        ("dpix2_resos", _p_f64),
        ("reso_redges", _p_f64),
        ("resoshift_leafs", ctypes.c_int32),
        ("minresoind_leaf", ctypes.c_int32),
        ("maxresoind_leaf", ctypes.c_int32),
        ("batch_membudget_mb", ctypes.c_int32),
    ]


class BinningParams(ctypes.Structure):
    """NPCF binning in real space (N=2) and multipole space (N>3).
    """
    _fields_ = [
        ("rmin", ctypes.c_double),
        ("rmax", ctypes.c_double),
        ("nbinsr", ctypes.c_int32),
        ("do_dc", ctypes.c_int32),
        ("nmax", ctypes.c_int32),
        ("nmin", ctypes.c_int32),
        ("dccorr", ctypes.c_int32),
        ("Pi", ctypes.c_double),
        ("rbins", _p_f64),
    ]

class FourthParams(ctypes.Structure):
    """NPCF binning for fourth-order stats.
    """
    _fields_ = [
        ("nbinsphi1", ctypes.c_int32),
        ("nbinsphi2", ctypes.c_int32),
        ("phibins1", _p_f64),
        ("phibins2", _p_f64),
        ("dbinsphi1", _p_f64),
        ("dbinsphi2", _p_f64),
        ("nindices", _p_i32),
        ("len_nindices", ctypes.c_int32),
        ("nthetacombis", ctypes.c_int32),
        ("nthetbatches", ctypes.c_int32),
        ("thetacombis_batches", _p_i32),
        ("nthetacombis_batches", _p_i32),
        ("cumthetacombis_batches",_p_i32),
    ]


class ClustCorr(ctypes.Structure):
    """Multiplicative clustering correction bundle for GNL.
    """
    _fields_ = [
        ("count_floor", ctypes.c_double),
        ("xi_nn", _p_f64),
        ("thetamin_xi", ctypes.c_double),
        ("thetamax_xi", ctypes.c_double),
        ("dtheta_xi", ctypes.c_double),
        ("has_xi", ctypes.c_int32),
        ("zeta", _p_f64),
        ("zeta_rbins", _p_f64),
        ("zeta_nr", ctypes.c_int32),
        ("zeta_phis", _p_f64),
        ("zeta_nphi", ctypes.c_int32),
        ("has_zeta", ctypes.c_int32),
    ]


class NPCFOutput(ctypes.Structure):
    """ Quantities that are returned by the clib functions.


    For each function the output can mean something slightly different:

    ======  =====================================  =========================
    NPCF    npcf (complex)                         norm
    ======  =====================================  =========================
    NN      NULL                                   ``norm``=weighted pairs,
                                                   ``npair_cell`` integer cells
    GG      [xip, xim] (ncomp=2)                   ``norm`` real, ``npair`` int
    NG      xi (ncomp=1)                           ``norm`` real, ``npair`` int
    GGG     Gammans (ncomp=4)                      ``norm_mp`` complex
    GNN/NGG Upsilon_n (ncomp=1/2)                  ``norm_mp`` complex
    ======  =====================================  =========================
    """
    _fields_ = [
        ("bin_centers", _p_f64),
        ("npcf", _p_c128),
        ("norm", _p_f64),
        ("norm_mp", _p_c128),
        ("npair", _p_i64),
        ("npair_cell", _p_i64),
        ("ncomp", ctypes.c_int32),
        ("nmax", ctypes.c_int32),
    ]


###################
# STRUCT BUILDERS #
###################

## (A) Catalog structs ##
def build_catalog_struct(mh, nbinsz, extra=None):
    """Populate a MultiresoCatalog structure.

    Parameters
    ----------
    mh : dict
        Bundle returned by ``Catalog.multihash_bundle()`` (``geometry`` in
        ``{'flat2d', 'spherical'}``).
    nbinsz : int
        Number of tomographic bins.
    extra : dict, optional
        Additional fields holding arrays from tracer catalogs.

    Returns
    -------
    (MultiresoCatalog, list)
        The struct and the list of arrays whose lifetime the caller must manage.

    Notes
    -----
    ``s.nresos`` is set to ``len(ngal_resos)``; for the flat tree hierarchy the
    caller should override it with ``self.tree_nresos`` (the flat bundle reports
    ``nresos = len(levels) - 1``, but the C rshift/loop convention counts every
    level).
    """
    geometry = mh['geometry']
    metric = 1 if geometry == 'spherical' else 0

    # Core arrays: for the spherical bundle these live under the red_* keys
    # (they act as the per-reduction galaxies for the C catalog struct).
    if geometry == 'spherical':
        isinner = np.ascontiguousarray(mh['red_isinner'], dtype=np.float64)
        weight  = np.ascontiguousarray(mh['red_w'], dtype=np.float64)
        zbin    = np.ascontiguousarray(mh['red_zbin'], dtype=np.int32)
    else:
        isinner = np.ascontiguousarray(mh['isinner_resos'], dtype=np.float64)
        weight  = np.ascontiguousarray(mh['weight_resos'], dtype=np.float64)
        zbin    = np.ascontiguousarray(mh['zbin_resos'], dtype=np.int32)
    ngal = np.ascontiguousarray(mh['ngal_resos'], dtype=np.int32)
    keepers = [isinner, weight, zbin, ngal]

    s = MultiresoCatalog()
    s.metric = metric
    s.nresos = int(len(ngal)) # This will be overridden when allocating tree parameters
    s.nbinsz = int(nbinsz)
    s.ngal_resos = _ptr_i32(ngal)
    s.isinner_resos = _ptr_f64(isinner)
    s.weight_resos = _ptr_f64(weight)
    s.zbin_resos = _ptr_i32(zbin)

    # Flat/3dbox transverse positional arrays (pos3 is the los coord, set by
    # the 3dbox slab builder only).
    s.pos1_resos = _null(_p_f64)
    s.pos2_resos = _null(_p_f64)
    s.pos3_resos = _null(_p_f64)
    if geometry == 'flat2d':
        pos1 = np.ascontiguousarray(mh['pos1_resos'], dtype=np.float64)
        pos2 = np.ascontiguousarray(mh['pos2_resos'], dtype=np.float64)
        keepers += [pos1, pos2]
        s.pos1_resos = _ptr_f64(pos1)
        s.pos2_resos = _ptr_f64(pos2)

    # Spherical-only unit-vector reduced positions (read as vx/vy/vz on the C side).
    s.vx_resos = _null(_p_f64)
    s.vy_resos = _null(_p_f64)
    s.vz_resos = _null(_p_f64)
    if geometry == 'spherical':
        vx = np.ascontiguousarray(mh['red_vx'], dtype=np.float64)
        vy = np.ascontiguousarray(mh['red_vy'], dtype=np.float64)
        vz = np.ascontiguousarray(mh['red_vz'], dtype=np.float64)
        keepers += [vx, vy, vz]
        s.vx_resos = _ptr_f64(vx)
        s.vy_resos = _ptr_f64(vy)
        s.vz_resos = _ptr_f64(vz)

    # Spherical-only bearing coordinates (ra, sin dec, cos dec); the spin-2
    # kernels feed these to sphere_bearing. Scalar NN leaves them NULL.
    s.ra_resos = _null(_p_f64)
    s.sindec_resos = _null(_p_f64)
    s.cosdec_resos = _null(_p_f64)
    if geometry == 'spherical' and 'red_ra' in mh:
        ra  = np.ascontiguousarray(mh['red_ra'],     dtype=np.float64)
        sd  = np.ascontiguousarray(mh['red_sindec'], dtype=np.float64)
        cd  = np.ascontiguousarray(mh['red_cosdec'], dtype=np.float64)
        keepers += [ra, sd, cd]
        s.ra_resos = _ptr_f64(ra)
        s.sindec_resos = _ptr_f64(sd)
        s.cosdec_resos = _ptr_f64(cd)

    # Shear-only arrays (NULL for NN).
    s.e1_resos = _null(_p_f64)
    s.e2_resos = _null(_p_f64)
    s.weightsq_resos = _null(_p_f64)
    if extra is not None:
        for name in ('e1_resos', 'e2_resos', 'weightsq_resos'):
            if name in extra:
                arr = np.ascontiguousarray(extra[name], dtype=np.float64)
                keepers.append(arr)
                setattr(s, name, _ptr_f64(arr))

    return s, keepers

def build_flat_catalog_struct(pos1, pos2, weight, zbin, nbinsz, isinner,
                              e1=None, e2=None, weightsq=None):
    """Single-resolution flat MultiresoCatalog from raw catalog arrays.

   This is a special case of MultiresoCatalog that is needed for the 
   discrete case and the tree-approximation.

    Returns
    -------
    (MultiresoCatalog, list)
    """
    p1 = np.ascontiguousarray(pos1, dtype=np.float64)
    p2 = np.ascontiguousarray(pos2, dtype=np.float64)
    w  = np.ascontiguousarray(weight, dtype=np.float64)
    zb = np.ascontiguousarray(zbin, dtype=np.int32)
    ii = np.ascontiguousarray(isinner, dtype=np.float64)
    ngal = np.array([len(p1)], dtype=np.int32)
    keepers = [p1, p2, w, zb, ii, ngal]

    s = MultiresoCatalog()
    s.metric = 0
    s.nresos = 1
    s.nbinsz = int(nbinsz)
    s.ngal_resos = _ptr_i32(ngal)
    s.isinner_resos = _ptr_f64(ii)
    s.weight_resos = _ptr_f64(w)
    s.zbin_resos = _ptr_i32(zb)
    s.pos1_resos = _ptr_f64(p1)
    s.pos2_resos = _ptr_f64(p2)
    for name, arr in (('e1_resos', e1), ('e2_resos', e2), ('weightsq_resos', weightsq)):
        if arr is not None:
            a = np.ascontiguousarray(arr, dtype=np.float64)
            keepers.append(a)
            setattr(s, name, _ptr_f64(a))

    return s, keepers

def build_slab_catalog_struct(mhs, nbinsz, e1e2=None):
    """Single-resolution MultiresoCatalog for 3dbox geometry with slab-hashing.

    Parameters
    ----------
    mhs : dict
         Bundle returned by ``Catalog.multihash_slabs``

    Returns
    -------
    (MultiresoCatalog, list)
    """
    p1 = np.ascontiguousarray(mhs['pos1'], dtype=np.float64)
    p2 = np.ascontiguousarray(mhs['pos2'], dtype=np.float64)
    p3 = np.ascontiguousarray(mhs['pos3'], dtype=np.float64)
    w = np.ascontiguousarray(mhs['weight'], dtype=np.float64)
    zb = np.ascontiguousarray(mhs['zbins'], dtype=np.int32)
    ii = np.ones(len(p1), dtype=np.float64)   # box has no buffer galaxies
    ngal = np.array([len(p1)], dtype=np.int32)
    keepers = [p1, p2, p3, w, zb, ii, ngal]

    s = MultiresoCatalog()
    s.metric = 2
    s.nresos = 1
    s.nbinsz = int(nbinsz)
    s.ngal_resos = _ptr_i32(ngal)
    s.isinner_resos = _ptr_f64(ii)
    s.weight_resos = _ptr_f64(w)
    s.zbin_resos = _ptr_i32(zb)
    s.pos1_resos = _ptr_f64(p1)
    s.pos2_resos = _ptr_f64(p2)
    s.pos3_resos = _ptr_f64(p3)
    if e1e2 is not None:
        e1 = np.ascontiguousarray(e1e2[0], dtype=np.float64)
        e2 = np.ascontiguousarray(e1e2[1], dtype=np.float64)
        keepers += [e1, e2]
        s.e1_resos = _ptr_f64(e1)
        s.e2_resos = _ptr_f64(e2)

    return s, keepers

## (B) Navhash structs ##
def build_navhash_struct(mh, cat_obj=None):
    """Populate a NavHash from a multihash bundle.

    Parameters
    ----------
    mh : dict
        Bundle returned by ``Catalog.multihash_bundle()``.
    cat_obj : Catalog, optional
        Original catalog; required on the flat path

    Returns
    -------
    (NavHash, list)
    """
    geometry = mh['geometry']
    metric = 1 if geometry == 'spherical' else 0
    keepers = []

    s = NavHash()
    s.metric = metric

    # 2dflat geometry
    s.index_matcher = _null(_p_i32)
    s.pixs_galind_bounds = _null(_p_i32)
    s.pix_gals = _null(_p_i32)
    s.index_matcher_hash = _null(_p_i32)
    s.filledregions = _null(_p_i32)
    s.nfilledregions = 0

    # 3dbox slab geometry (this feeds into build_slab_navhash_struct)
    s.slab_offsets = _null(_p_i32)
    s.rshift_bounds = _null(_p_i32)
    s.nslabs = 0
    s.z0 = 0.
    s.dpix_z = 0.

    if geometry == 'flat2d':
        assert cat_obj is not None, "cat_obj required for flat NavHash"
        im  = np.ascontiguousarray(mh['index_matcher_resos'],      dtype=np.int32)
        pgb = np.ascontiguousarray(mh['pixs_galind_bounds_resos'], dtype=np.int32)
        pg  = np.ascontiguousarray(mh['pix_gals_resos'],           dtype=np.int32)
        # Indices of occupied cells in base-resolution hash 
        imh = np.ascontiguousarray(
            np.argwhere(cat_obj.index_matcher > -1).flatten(), dtype=np.int32)
        nregions = len(imh)
        # In C we only want to iterate over regions that contain at least one
        # inner galaxy. Here we set up the array to efficiently enumerate through this set
        base_bounds = np.asarray(cat_obj.pixs_galind_bounds)
        inner_ingal = np.asarray(cat_obj.isinner, dtype=np.float64)[np.asarray(cat_obj.pix_gals)]
        inner_csum = np.concatenate(([0.], np.cumsum(inner_ingal)))
        inner_per_cell = inner_csum[base_bounds[1:nregions+1]] - inner_csum[base_bounds[:nregions]]
        filled = np.ascontiguousarray(np.nonzero(inner_per_cell > 0)[0].astype(np.int32))
        keepers += [im, pgb, pg, imh, filled]

        s.index_matcher = _ptr_i32(im)
        s.pixs_galind_bounds = _ptr_i32(pgb)
        s.pix_gals = _ptr_i32(pg)
        s.pix1_start = float(cat_obj.pix1_start)
        s.pix1_d = float(cat_obj.pix1_d)
        s.pix1_n = int(cat_obj.pix1_n)
        s.pix2_start = float(cat_obj.pix2_start)
        s.pix2_d = float(cat_obj.pix2_d)
        s.pix2_n = int(cat_obj.pix2_n)
        s.nregions = nregions
        s.index_matcher_hash = _ptr_i32(imh)
        s.filledregions = _ptr_i32(filled)
        s.nfilledregions = len(filled)

    # Spherical geometry
    s.ncells_resos = _null(_p_i32)
    s.nside_nav = _null(_p_long)
    s.cell_pix = _null(_p_long)
    s.cell_redbounds = _null(_p_i32)
    s.rshift_red = _null(_p_i32)
    s.rshift_cellpix = _null(_p_i32)
    s.rshift_cellbounds = _null(_p_i32)

    if geometry == 'spherical':
        ncr  = np.ascontiguousarray(mh['ncells_resos'], dtype=np.int32)
        nsn  = np.ascontiguousarray(mh['nside_nav'], dtype=np.int64)
        cp   = np.ascontiguousarray(mh['cell_pix'], dtype=np.int64)
        clb  = np.ascontiguousarray(mh['cell_redbounds'], dtype=np.int32)
        rsl  = np.ascontiguousarray(mh['rshift_red'], dtype=np.int32)
        rscp = np.ascontiguousarray(mh['rshift_cellpix'], dtype=np.int32)
        rscb = np.ascontiguousarray(mh['rshift_cellbounds'], dtype=np.int32)
        keepers += [ncr, nsn, cp, clb, rsl, rscp, rscb]

        s.ncells_resos = _ptr_i32(ncr)
        s.nside_nav = _ptr_long(nsn)
        s.cell_pix = _ptr_long(cp)
        s.cell_redbounds = _ptr_i32(clb)
        s.rshift_red = _ptr_i32(rsl)
        s.rshift_cellpix = _ptr_i32(rscp)
        s.rshift_cellbounds = _ptr_i32(rscb)

    return s, keepers

def build_flat_navhash_struct(cat_obj):
    """Single-resolution flat NavHash from a catalog's base spatial hash.

    This is a special case of NavHash that is needed for the discrete
    case and the tree-approximation.

    Parameters
    ----------
    cat_obj : Catalog
        Catalog object containing the hashes

    Returns
    -------
    (NavHash, list)
    """
    im  = np.ascontiguousarray(cat_obj.index_matcher, dtype=np.int32)
    pgb = np.ascontiguousarray(cat_obj.pixs_galind_bounds, dtype=np.int32)
    pg  = np.ascontiguousarray(cat_obj.pix_gals, dtype=np.int32)
    # In C we only want to iterate over regions that contain at least one
    # inner galaxy. Here we set up the array to efficiently enumerate through this set
    nregions = int(np.count_nonzero(im > -1))
    inner_ingal = np.asarray(cat_obj.isinner, dtype=np.float64)[pg]
    inner_csum = np.concatenate(([0.], np.cumsum(inner_ingal)))
    inner_per_cell = inner_csum[pgb[1:nregions+1]] - inner_csum[pgb[:nregions]]
    filled = np.ascontiguousarray(np.nonzero(inner_per_cell > 0)[0].astype(np.int32))
    keepers = [im, pgb, pg, filled]

    s = NavHash()
    s.metric = 0
    s.index_matcher = _ptr_i32(im)
    s.pixs_galind_bounds = _ptr_i32(pgb)
    s.pix_gals = _ptr_i32(pg)
    s.pix1_start = float(cat_obj.pix1_start)
    s.pix1_d = float(cat_obj.pix1_d)
    s.pix1_n = int(cat_obj.pix1_n)
    s.pix2_start = float(cat_obj.pix2_start)
    s.pix2_d = float(cat_obj.pix2_d)
    s.pix2_n = int(cat_obj.pix2_n)
    s.nregions = nregions
    s.filledregions = _ptr_i32(filled)
    s.nfilledregions = len(filled)

    return s, keepers


def build_slab_navhash_struct(mhs):
    """Populate a NavHash from a multihash slab bundle.

    Parameters
    ----------
    mhs : dict
         Bundle returned by ``Catalog.multihash_slabs``

    Returns
    -------
    (NavHash, list)
    """
    im  = np.ascontiguousarray(mhs['index_matcher'], dtype=np.int32)
    pgb = np.ascontiguousarray(mhs['pixs_galind_bounds'], dtype=np.int32)
    pg  = np.ascontiguousarray(mhs['pix_gals'], dtype=np.int32)
    so  = np.ascontiguousarray(mhs['slab_offsets'], dtype=np.int32)
    rsb = np.ascontiguousarray(mhs['rshift_bounds'], dtype=np.int32)
    keepers = [im, pgb, pg, so, rsb]

    s = NavHash()
    s.metric = 2
    s.index_matcher = _ptr_i32(im)
    s.pixs_galind_bounds = _ptr_i32(pgb)
    s.pix_gals = _ptr_i32(pg)
    s.pix1_start = float(mhs['pix1_start'])
    s.pix1_d = float(mhs['pix1_d'])
    s.pix1_n = int(mhs['pix1_n'])
    s.pix2_start = float(mhs['pix2_start'])
    s.pix2_d = float(mhs['pix2_d'])
    s.pix2_n = int(mhs['pix2_n'])
    s.slab_offsets = _ptr_i32(so)
    s.rshift_bounds = _ptr_i32(rsb)
    s.nslabs = int(mhs['nslabs'])
    s.z0 = float(mhs['z0'])
    s.dpix_z = float(mhs['dpix_z'])

    return s, keepers

## (C) Other structs ##
def build_tree_params_struct(corr, mh):
    """Populate a TreeResoParams from a correlator instance and a bundle.

    Parameters
    ----------
    corr: BinnedNPCF
         ``BinnedNPCF`` object containing the parameters for the 
         tree-based approximations
    mh : dict
         Bundle returned by ``Catalog.multihash_bundle``

    Returns
    -------
    (TreeResoParams, list)
    """
    keepers = []
    geometry = mh['geometry']

    if geometry == 'flat2d':
        dp1 = np.ascontiguousarray(mh['dpixs1_true'], dtype=np.float64)
        dp2 = np.ascontiguousarray(mh['dpixs2_true'], dtype=np.float64)
        re  = np.ascontiguousarray(corr.tree_redges, dtype=np.float64)
    else:
        dp1 = np.zeros(0, dtype=np.float64)
        dp2 = np.zeros(0, dtype=np.float64)
        re  = np.ascontiguousarray(mh['reso_redges'], dtype=np.float64)
    keepers += [dp1, dp2, re]

    s = TreeResoParams()
    s.nresos = int(corr.tree_nresos)
    s.nresos_grid = int(corr.tree_nresos)
    s.dpix1_resos = _ptr_f64(dp1)
    s.dpix2_resos = _ptr_f64(dp2)
    s.reso_redges = _ptr_f64(re)
    s.resoshift_leafs = int(corr.resoshift_leafs)
    s.minresoind_leaf = int(corr.minresoind_leaf)
    s.maxresoind_leaf = int(corr.maxresoind_leaf)
    s.batch_membudget_mb = int(corr.batch_membudget_mb)

    return s, keepers


def build_binning_struct(corr, scale=None, do_dc=None, nmax=0, nmin=0, dccorr=0,
                         Pi=0., rbins=None):
    """Populate a BinningParams from a correlator instance.

    Parameters
    ----------
    corr : BinnedNPCF instance
    scale : float or None
        Factor converting ``min_sep``/``max_sep`` from the correlator's
        ``sep_units`` into the working unit passed to C (radians on the spherical
        path). ``None`` leaves the values in ``sep_units`` (flat path).
    do_dc : bool or None
        Whether to explicitly double count pairs. Only used for 2PCFs.
    nmax, nmin : int
        Multipole order range (GGG and higher); ignored by NN/GG.
    dccorr : int
        Multi-count correction toggle (GGG); ignored by NN/GG.
    Pi : float
        Line-of-sight window half-width. Only used for 3dbox metric.
    rbins : ndarray or None
        Explicit log-r bin edges (GGG discrete). ``None`` leaves the field NULL
        (the C side recomputes them from rmin/rmax/nbinsr).

    Returns
    -------
    BinningParams
        The array-owning ``rbins`` case stashes the keep-alive array on the
        returned struct (``s._rbins_keep``) so no separate keeper list is needed.
    """
    s = BinningParams()
    f = scale if scale is not None else 1.0
    s.rmin = float(corr.min_sep) * f
    s.rmax = float(corr.max_sep) * f
    s.nbinsr = int(corr.nbinsr)
    s.do_dc = int(1) if do_dc is None else int(do_dc)
    s.nmax = int(nmax)
    s.nmin = int(nmin)
    s.dccorr = int(dccorr)
    s.Pi = float(Pi)
    s.rbins  = _null(_p_f64)
    if rbins is not None:
        rb = np.ascontiguousarray(rbins, dtype=np.float64)
        s._rbins_keep = rb
        s.rbins = _ptr_f64(rb)
    return s


def build_npcf_output(kind, nbinsr, nmax=0, nbinsz=None, nbinsz_lens=None,
                      nbinsz_source=None, estimator_type='standard'):
    """Initialise output arrays and wrap them in a unified NPCFOutput.

    Parameters
    ----------
    kind : {'NN', 'GG', 'NG', 'GGG', 'NNN', 'GNN', 'NGG'}
        Type of correlator family.
    nbinsr : int
        The number of radial bins
    nmax : int
        Multipole order (only relevant for orders >2).
    nbinsz, nbinsz_lens, nbinsz_source : int
        Tomographic bin counts (per kind: NN/GG/GGG use ``nbinsz``; NG uses
        lens+source).
    estimator_type : {'standard', 'lslike_slab'}
        Output parametrisation for the 3pt families. ``'standard'`` emits the
        natural components. ``'lslike_slab'`` emits the Landy-Szalay-like stack
        of correlators when randoms are included in the estimation.

    Returns
    -------
    (NPCFOutput, bin_centers, npcf, norm, norm_mp, npair, npair_cell)
        Unused arrays are ``None``.
    """
    s = NPCFOutput()
    s.nmax = int(nmax)
    bin_centers = npcf = norm = norm_mp = npair = npair_cell = None

    if kind == 'NN':
        s.ncomp = 0
        z2r = nbinsz*nbinsz*nbinsr
        bin_centers = np.zeros(z2r, dtype=np.float64)
        norm = np.zeros(z2r, dtype=np.float64)
        npair_cell = np.zeros(z2r, dtype=np.int64)
        s.norm = _ptr_f64(norm)
        s.npair_cell = _ptr_i64(npair_cell)
    elif kind == 'GG':
        s.ncomp = 2
        z2r = nbinsz*nbinsz*nbinsr
        bin_centers = np.zeros(z2r, dtype=np.float64)
        npcf = np.zeros(s.ncomp*z2r, dtype=np.complex128)
        norm = np.zeros(z2r, dtype=np.float64)
        npair = np.zeros(z2r, dtype=np.int64)
        s.npcf = npcf.ctypes.data_as(_p_c128)
        s.norm = _ptr_f64(norm)
        s.npair = _ptr_i64(npair)
    elif kind == 'NG':
        s.ncomp = 1
        z2r = nbinsz_lens*nbinsz_source*nbinsr
        bin_centers = np.zeros(z2r, dtype=np.float64)
        npcf = np.zeros(s.ncomp*z2r, dtype=np.complex128)
        norm = np.zeros(z2r, dtype=np.float64)
        npair = np.zeros(z2r, dtype=np.int64)
        s.npcf = npcf.ctypes.data_as(_p_c128)
        s.norm = _ptr_f64(norm)
        s.npair = _ptr_i64(npair)
    elif kind == 'GGG':
        s.ncomp = 4
        nzc = nbinsz*nbinsz*nbinsz
        comp = (nmax+1)*nzc*nbinsr*nbinsr
        nbc = nbinsz*nbinsz if estimator_type == 'lslike_slab' else nbinsz
        bin_centers = np.zeros(nbc*nbinsr, dtype=np.float64)
        npcf = np.zeros(s.ncomp*comp, dtype=np.complex128)
        norm_mp = np.zeros(comp, dtype=np.complex128)
        s.npcf = npcf.ctypes.data_as(_p_c128)
        s.norm_mp = norm_mp.ctypes.data_as(_p_c128)
    elif kind == 'NNN':
        s.ncomp = 1
        nzc = nbinsz*nbinsz*nbinsz
        comp = (nmax+1)*nzc*nbinsr*nbinsr
        bin_centers = np.zeros(nbinsz*nbinsr, dtype=np.float64)
        npcf = np.zeros(s.ncomp*comp, dtype=np.complex128)
        norm_mp = np.zeros(comp, dtype=np.complex128)
        s.npcf = npcf.ctypes.data_as(_p_c128)
        s.norm_mp = norm_mp.ctypes.data_as(_p_c128)
    elif kind == 'GNN':
        s.ncomp = 1
        nzc = nbinsz_source*nbinsz_lens*nbinsz_lens
        comp = (nmax+1)*nzc*nbinsr*nbinsr
        stack = 4 if estimator_type == 'lslike_slab' else 1 
        bin_centers = np.zeros(nbinsz_lens*nbinsz_source*nbinsr, dtype=np.float64)
        npcf = np.zeros(stack*s.ncomp*comp, dtype=np.complex128)
        norm_mp = np.zeros(comp, dtype=np.complex128)
        s.npcf = npcf.ctypes.data_as(_p_c128)
        s.norm_mp = norm_mp.ctypes.data_as(_p_c128)
    elif kind == 'NGG':
        s.ncomp = 2
        nzc = nbinsz_lens*nbinsz_source*nbinsz_source
        comp = (2*nmax+1)*nzc*nbinsr*nbinsr
        stack = 2 if estimator_type == 'lslike_slab' else 1
        bin_centers = np.zeros(nbinsz_lens*nbinsz_source*nbinsr, dtype=np.float64)
        npcf = np.zeros(stack*s.ncomp*comp, dtype=np.complex128)
        norm_mp = np.zeros(comp, dtype=np.complex128)
        s.npcf = npcf.ctypes.data_as(_p_c128)
        s.norm_mp = norm_mp.ctypes.data_as(_p_c128)
    else:
        raise ValueError(f"unknown NPCFOutput kind {kind!r}")

    s.bin_centers = _ptr_f64(bin_centers)
    return s, bin_centers, npcf, norm, norm_mp, npair, npair_cell


## (D) Fourth-order structs ##
def build_gnnn_output(bin_centers, Gtilde_n, N_n):
    """NPCFOutput wrapping pre-allocated GNNN 4pcf-multipole arrays.
    """
    s = NPCFOutput()
    s.ncomp = 1
    s.bin_centers = _ptr_f64(bin_centers)
    s.npcf = Gtilde_n.ctypes.data_as(_p_c128)
    s.norm_mp = N_n.ctypes.data_as(_p_c128)
    return s


def build_gggg_output(bin_centers, Upsilon_n, N_n):
    """NPCFOutput wrapping pre-allocated GGGG 4pcf-multipole arrays (8 components
    packed in the flat npcf buffer).
    """
    s = NPCFOutput()
    s.ncomp = 8
    s.bin_centers = _ptr_f64(bin_centers)
    s.npcf = Upsilon_n.ctypes.data_as(_p_c128)
    s.norm_mp = N_n.ctypes.data_as(_p_c128)
    return s


def build_nnnn_output(bin_centers, N_n):
    """NPCFOutput wrapping the pre-allocated NNNN 4pcf-count multipoles.
    """
    s = NPCFOutput()
    s.ncomp = 1
    s.bin_centers = _ptr_f64(bin_centers)
    s.npcf = N_n.ctypes.data_as(_p_c128)
    return s


def build_spherical_central_catalog_struct(isinner, weight, vx, vy, vz, ra, sindec, cosdec, nbinsz):
    """Single-resolution spherical MultiresoCatalog holding the full-resolution
    central (query) galaxies of the curved-sky NNNN estimator.
    """
    ii = np.ascontiguousarray(isinner, dtype=np.float64)
    w  = np.ascontiguousarray(weight, dtype=np.float64)
    vx = np.ascontiguousarray(vx, dtype=np.float64)
    vy = np.ascontiguousarray(vy, dtype=np.float64)
    vz = np.ascontiguousarray(vz, dtype=np.float64)
    ra = np.ascontiguousarray(ra, dtype=np.float64)
    sd = np.ascontiguousarray(sindec, dtype=np.float64)
    cd = np.ascontiguousarray(cosdec, dtype=np.float64)
    ngal = np.array([len(ii)], dtype=np.int32)
    keepers = [ii, w, vx, vy, vz, ra, sd, cd, ngal]

    s = MultiresoCatalog()
    s.metric = 1
    s.nresos = 1
    s.nbinsz = int(nbinsz)
    s.ngal_resos = _ptr_i32(ngal)
    s.isinner_resos = _ptr_f64(ii)
    s.weight_resos = _ptr_f64(w)
    s.vx_resos = _ptr_f64(vx)
    s.vy_resos = _ptr_f64(vy)
    s.vz_resos = _ptr_f64(vz)
    s.ra_resos = _ptr_f64(ra)
    s.sindec_resos = _ptr_f64(sd)
    s.cosdec_resos = _ptr_f64(cd)
    return s, keepers


def build_fourth_params(nindices=None, len_nindices=0, nthetacombis=0, nthetbatches=0,
                        phibins1=None, phibins2=None, dbinsphi1=None, dbinsphi2=None,
                        nbinsphi1=0, nbinsphi2=0,
                        thetacombis_batches=None, nthetacombis_batches=None,
                        cumthetacombis_batches=None):
    """Populate a FourthParams struct. 

    Returns
    -------
    (FourthParams, list)
    """
    keepers = []
    s = FourthParams()
    s.nbinsphi1 = int(nbinsphi1)
    s.nbinsphi2 = int(nbinsphi2)
    s.len_nindices = int(len_nindices)
    s.nthetacombis = int(nthetacombis)
    s.nthetbatches = int(nthetbatches)
    for name, arr, dt in (('phibins1', phibins1, np.float64), ('phibins2', phibins2, np.float64),
                          ('dbinsphi1', dbinsphi1, np.float64), ('dbinsphi2', dbinsphi2, np.float64),
                          ('nindices', nindices, np.int32),
                          ('thetacombis_batches', thetacombis_batches, np.int32),
                          ('nthetacombis_batches', nthetacombis_batches, np.int32),
                          ('cumthetacombis_batches', cumthetacombis_batches, np.int32)):
        if arr is None:
            setattr(s, name, _null(_p_i32 if dt is np.int32 else _p_f64))
        else:
            a = np.ascontiguousarray(arr, dtype=dt)
            keepers.append(a)
            setattr(s, name, (_ptr_i32 if dt is np.int32 else _ptr_f64)(a))
    return s, keepers


def build_clustcorr(corr, xi, nnn, count_floor):
    """Populate a ClustCorr from the correlator's ``_clustcorr_args`` block.

    Parameters
    ----------
    corr : GNNNCorrelation_NoTomo
        Provides ``_clustcorr_args(xi, nnn)`` (the omega/zeta resampling logic).
    xi : tuple or None
        Angular clustering 2PCF ``(thetas, omega)`` of the lenses.
    nnn : orpheus.NNNCorrelation or tuple or None
        Connected lens 3PCF for the zeta term.
    count_floor : float
        Threshold on the reconstructed triplet counts below which the 4PCF is zeroed.

    Returns
    -------
    (ClustCorr, list)
    """
    (xi_nn, thetamin_xi, thetamax_xi, dtheta_xi, has_xi,
     zeta, zeta_rbins, zeta_nr, zeta_phis, zeta_nphi, has_zeta) = corr._clustcorr_args(xi, nnn)
    xi_nn = np.ascontiguousarray(xi_nn, dtype=np.float64)
    zeta = np.ascontiguousarray(zeta, dtype=np.float64)
    zeta_rbins = np.ascontiguousarray(zeta_rbins, dtype=np.float64)
    zeta_phis = np.ascontiguousarray(zeta_phis, dtype=np.float64)
    keepers = [xi_nn, zeta, zeta_rbins, zeta_phis]

    s = ClustCorr()
    s.count_floor = float(count_floor)
    s.xi_nn = _ptr_f64(xi_nn)
    s.thetamin_xi = float(thetamin_xi)
    s.thetamax_xi = float(thetamax_xi)
    s.dtheta_xi = float(dtheta_xi)
    s.has_xi = int(has_xi)
    s.zeta = _ptr_f64(zeta)
    s.zeta_rbins = _ptr_f64(zeta_rbins)
    s.zeta_nr = int(zeta_nr)
    s.zeta_phis = _ptr_f64(zeta_phis)
    s.zeta_nphi = int(zeta_nphi)
    s.has_zeta = int(has_zeta)
    return s, keepers
