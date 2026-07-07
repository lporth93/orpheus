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
        ("metric",          ctypes.c_int32),
        ("nresos",          ctypes.c_int32),
        ("ngal_resos",      _p_i32),
        ("nbinsz",          ctypes.c_int32),
        ("isinner_resos",   _p_f64),
        ("weight_resos",    _p_f64),
        ("zbin_resos",      _p_i32),
        # Cartesian position coordinate (used for 2d and 3d flat methods)
        ("pos1_resos",      _p_f64),
        ("pos2_resos",      _p_f64),
        ("pos3_resos",      _p_f64),
        # 3D-vectors used for spherical ra/dec coordinates (used for sphierical geometries)
        ("vx_resos",        _p_f64),
        ("vy_resos",        _p_f64),
        ("vz_resos",        _p_f64),
        # Helper quantities for the spin-2 geodesic bearing (used for sphierical geometries)
        ("ra_resos",        _p_f64),
        ("sindec_resos",    _p_f64),
        ("cosdec_resos",    _p_f64),
        # Spin-2 tracer quantities (Used for SpinTracerCatalogs)
        ("e1_resos",        _p_f64),
        ("e2_resos",        _p_f64),
        ("weightsq_resos",  _p_f64),
    ]


class NavHash(ctypes.Structure):
    """Quantities neccessary to navigate through a multihash nested spatial hashing structure
    """
    _fields_ = [
        ("metric",              ctypes.c_int32),
        # Relevant for flat2d metric
        ("index_matcher",       _p_i32),
        ("pixs_galind_bounds",  _p_i32),
        ("pix_gals",            _p_i32),
        ("pix1_start",          ctypes.c_double),
        ("pix1_d",              ctypes.c_double),
        ("pix1_n",              ctypes.c_int32),
        ("pix2_start",          ctypes.c_double),
        ("pix2_d",              ctypes.c_double),
        ("pix2_n",              ctypes.c_int32),
        ("nregions",            ctypes.c_int32),
        ("index_matcher_hash",  _p_i32),
        ("filledregions",       _p_i32),
        ("nfilledregions",      ctypes.c_int32),
        # Additionally relevant for 3dbox ensemble of multihashes
        ("slab_offsets",        _p_i32),
        ("rshift_bounds",       _p_i32),
        ("nslabs",              ctypes.c_int32),
        ("z0",                  ctypes.c_double),
        ("dpix_z",              ctypes.c_double),
        # Relevant for spherical metric
        ("ncells_resos",        _p_i32),
        ("nside_nav",           _p_long),
        ("cell_pix",            _p_long),
        ("cell_redbounds",      _p_i32),
        ("rshift_red",          _p_i32),
        ("rshift_cellpix",      _p_i32),
        ("rshift_cellbounds",   _p_i32),
    ]


class TreeResoParams(ctypes.Structure):
    """Quantities neccessary to describe the resolution levels of the multihashes
    """
    _fields_ = [
        ("nresos",           ctypes.c_int32),
        ("nresos_grid",      ctypes.c_int32),
        ("dpix1_resos",      _p_f64),
        ("dpix2_resos",      _p_f64),
        ("reso_redges",      _p_f64),
        ("resoshift_leafs",  ctypes.c_int32),
        ("minresoind_leaf",  ctypes.c_int32),
        ("maxresoind_leaf",  ctypes.c_int32),
    ]


class BinningParams(ctypes.Structure):
    """NPCF binning in real space (N=2) and multipole space (N>3).
    """
    _fields_ = [
        ("rmin",    ctypes.c_double),
        ("rmax",    ctypes.c_double),
        ("nbinsr",  ctypes.c_int32),
        ("do_dc",   ctypes.c_int32),
        ("nmax",    ctypes.c_int32),
        ("nmin",    ctypes.c_int32),
        ("dccorr",  ctypes.c_int32),
        ("Pi",      ctypes.c_double),
        ("rbins",   _p_f64),
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
        ("bin_centers",  _p_f64),
        ("npcf",         _p_c128),
        ("norm",         _p_f64),
        ("norm_mp",      _p_c128),
        ("npair",        _p_i64),
        ("npair_cell",   _p_i64),
        ("ncomp",        ctypes.c_int32),
        ("nmax",         ctypes.c_int32),
    ]


###################
# STRUCT BUILDERS #
###################

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
        weight  = np.ascontiguousarray(mh['red_w'],       dtype=np.float64)
        zbin    = np.ascontiguousarray(mh['red_zbin'],    dtype=np.int32)
    else:
        isinner = np.ascontiguousarray(mh['isinner_resos'], dtype=np.float64)
        weight  = np.ascontiguousarray(mh['weight_resos'],  dtype=np.float64)
        zbin    = np.ascontiguousarray(mh['zbin_resos'],    dtype=np.int32)
    ngal = np.ascontiguousarray(mh['ngal_resos'], dtype=np.int32)
    keepers = [isinner, weight, zbin, ngal]

    s = MultiresoCatalog()
    s.metric     = metric
    s.nresos     = int(len(ngal))
    s.nbinsz     = int(nbinsz)
    s.ngal_resos = _ptr_i32(ngal)
    s.isinner_resos = _ptr_f64(isinner)
    s.weight_resos  = _ptr_f64(weight)
    s.zbin_resos    = _ptr_i32(zbin)

    # Flat / 3dbox transverse positional arrays (pos3 is the LOS coord, set by
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
    s.ra_resos     = _null(_p_f64)
    s.sindec_resos = _null(_p_f64)
    s.cosdec_resos = _null(_p_f64)
    if geometry == 'spherical' and 'red_ra' in mh:
        ra  = np.ascontiguousarray(mh['red_ra'],     dtype=np.float64)
        sd  = np.ascontiguousarray(mh['red_sindec'], dtype=np.float64)
        cd  = np.ascontiguousarray(mh['red_cosdec'], dtype=np.float64)
        keepers += [ra, sd, cd]
        s.ra_resos     = _ptr_f64(ra)
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


def build_navhash_struct(mh, cat_obj=None):
    """Populate a NavHash from a multihash bundle.

    Parameters
    ----------
    mh : dict
        Bundle returned by ``Catalog.multihash_bundle()``.
    cat_obj : Catalog, optional
        Original catalog; required on the flat path (for the hash grid geometry
        and to build ``filledregions``/``index_matcher_hash``).

    Returns
    -------
    (NavHash, list)
    """
    geometry = mh['geometry']
    metric = 1 if geometry == 'spherical' else 0
    keepers = []

    s = NavHash()
    s.metric = metric

    # --- flat ---
    s.index_matcher      = _null(_p_i32)
    s.pixs_galind_bounds = _null(_p_i32)
    s.pix_gals           = _null(_p_i32)
    s.index_matcher_hash = _null(_p_i32)
    s.filledregions      = _null(_p_i32)
    s.nfilledregions     = 0

    # --- 3dbox slab (populated by build_slab_navhash_struct) ---
    s.slab_offsets  = _null(_p_i32)
    s.rshift_bounds = _null(_p_i32)
    s.nslabs = 0
    s.z0     = 0.
    s.dpix_z = 0.

    if geometry == 'flat2d':
        assert cat_obj is not None, "cat_obj required for flat NavHash"
        im  = np.ascontiguousarray(mh['index_matcher_resos'],      dtype=np.int32)
        pgb = np.ascontiguousarray(mh['pixs_galind_bounds_resos'], dtype=np.int32)
        pg  = np.ascontiguousarray(mh['pix_gals_resos'],           dtype=np.int32)
        # Occupied-cell slots of the base-resolution hash (nregions = #occupied
        # cells), mirroring the flat process() convention.
        imflat = np.ascontiguousarray(
            np.argwhere(cat_obj.index_matcher > -1).flatten(), dtype=np.int32)
        nregions = len(imflat)
        # Iterate only regions with >=1 inner galaxy (mirrors the pre-refactor
        # flat convention). Buffer-only cells contribute exactly zero but would
        # still pay per-region cache setup + cross-reso accumulation, so skip
        # them -- matters for padded/patch catalogs, no-op when all-inner.
        base_bounds = np.asarray(cat_obj.pixs_galind_bounds)
        inner_ingal = np.asarray(cat_obj.isinner, dtype=np.float64)[np.asarray(cat_obj.pix_gals)]
        inner_csum = np.concatenate(([0.], np.cumsum(inner_ingal)))
        inner_per_cell = inner_csum[base_bounds[1:nregions+1]] - inner_csum[base_bounds[:nregions]]
        filled = np.ascontiguousarray(np.nonzero(inner_per_cell > 0)[0].astype(np.int32))
        keepers += [im, pgb, pg, imflat, filled]

        s.index_matcher      = _ptr_i32(im)
        s.pixs_galind_bounds = _ptr_i32(pgb)
        s.pix_gals           = _ptr_i32(pg)
        s.pix1_start = float(cat_obj.pix1_start)
        s.pix1_d     = float(cat_obj.pix1_d)
        s.pix1_n     = int(cat_obj.pix1_n)
        s.pix2_start = float(cat_obj.pix2_start)
        s.pix2_d     = float(cat_obj.pix2_d)
        s.pix2_n     = int(cat_obj.pix2_n)
        s.nregions   = nregions
        s.index_matcher_hash = _ptr_i32(imflat)
        s.filledregions  = _ptr_i32(filled)
        s.nfilledregions = len(filled)

    # --- spherical ---
    s.ncells_resos   = _null(_p_i32)
    s.nside_nav      = _null(_p_long)
    s.cell_pix       = _null(_p_long)
    s.cell_redbounds = _null(_p_i32)
    s.rshift_red     = _null(_p_i32)
    s.rshift_cellpix = _null(_p_i32)
    s.rshift_cellbounds = _null(_p_i32)

    if geometry == 'spherical':
        ncr  = np.ascontiguousarray(mh['ncells_resos'],      dtype=np.int32)
        nsn  = np.ascontiguousarray(mh['nside_nav'],         dtype=np.int64)
        cp   = np.ascontiguousarray(mh['cell_pix'],          dtype=np.int64)
        clb  = np.ascontiguousarray(mh['cell_redbounds'],    dtype=np.int32)
        rsl  = np.ascontiguousarray(mh['rshift_red'],        dtype=np.int32)
        rscp = np.ascontiguousarray(mh['rshift_cellpix'],    dtype=np.int32)
        rscb = np.ascontiguousarray(mh['rshift_cellbounds'], dtype=np.int32)
        keepers += [ncr, nsn, cp, clb, rsl, rscp, rscb]

        # C long and int64 are both 64-bit on the target platforms.
        s.ncells_resos   = _ptr_i32(ncr)
        s.nside_nav      = _ptr_long(nsn)
        s.cell_pix       = _ptr_long(cp)
        s.cell_redbounds = _ptr_i32(clb)
        s.rshift_red     = _ptr_i32(rsl)
        s.rshift_cellpix = _ptr_i32(rscp)
        s.rshift_cellbounds = _ptr_i32(rscb)

    return s, keepers


def build_flat_catalog_struct(pos1, pos2, weight, zbin, nbinsz, isinner,
                              e1=None, e2=None, weightsq=None):
    """Single-resolution (nresos=1) flat MultiresoCatalog from raw catalog arrays.

    The discrete / tree-base 'query' catalog: the finest-resolution galaxies fed
    to a kernel as reso 0. Shear arrays (e1/e2/weightsq) are optional -- scalar
    catalogs leave them NULL. Mirrors build_catalog_struct's field layout without
    a multihash bundle.

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
    s.metric        = 0
    s.nresos        = 1
    s.nbinsz        = int(nbinsz)
    s.ngal_resos    = _ptr_i32(ngal)
    s.isinner_resos = _ptr_f64(ii)
    s.weight_resos  = _ptr_f64(w)
    s.zbin_resos    = _ptr_i32(zb)
    s.pos1_resos    = _ptr_f64(p1)
    s.pos2_resos    = _ptr_f64(p2)
    for name, arr in (('e1_resos', e1), ('e2_resos', e2), ('weightsq_resos', weightsq)):
        if arr is not None:
            a = np.ascontiguousarray(arr, dtype=np.float64)
            keepers.append(a)
            setattr(s, name, _ptr_f64(a))

    return s, keepers


def build_flat_navhash_struct(cat_obj):
    """Single-resolution flat NavHash from a catalog's base spatial hash.

    The discrete-path navigation: the base-resolution pixel hash already on the
    catalog (``index_matcher`` / ``pixs_galind_bounds`` / ``pix_gals`` + grid
    geometry). ``nregions`` = number of occupied cells (used by the discrete G3L
    kernels). No multi-resolution region list / index_matcher_hash (only needed by
    BaseTree/DoubleTree, which use build_navhash_struct).

    Returns
    -------
    (NavHash, list)
    """
    im  = np.ascontiguousarray(cat_obj.index_matcher,      dtype=np.int32)
    pgb = np.ascontiguousarray(cat_obj.pixs_galind_bounds, dtype=np.int32)
    pg  = np.ascontiguousarray(cat_obj.pix_gals,           dtype=np.int32)
    keepers = [im, pgb, pg]

    s = NavHash()
    s.metric             = 0
    s.index_matcher      = _ptr_i32(im)
    s.pixs_galind_bounds = _ptr_i32(pgb)
    s.pix_gals           = _ptr_i32(pg)
    s.pix1_start = float(cat_obj.pix1_start)
    s.pix1_d     = float(cat_obj.pix1_d)
    s.pix1_n     = int(cat_obj.pix1_n)
    s.pix2_start = float(cat_obj.pix2_start)
    s.pix2_d     = float(cat_obj.pix2_d)
    s.pix2_n     = int(cat_obj.pix2_n)
    s.nregions   = int(len(np.argwhere(im > -1)))

    return s, keepers


def build_slab_catalog_struct(sh, nbinsz, e1e2=None):
    """Single-resolution (nresos=1) '3dbox' MultiresoCatalog from a slab hash.

    ``sh`` is a ``Catalog.multihash_slabs`` bundle (its ``pos1/pos2/pos3/weight/
    zbins`` are already reordered so each line-of-sight slab is contiguous). The
    LOS coordinate goes to ``pos3_resos``; a polar leg additionally passes its
    reordered ``(e1, e2)`` (the bundle's ``'fields'``) via ``e1e2``. Scalar
    catalogs leave e1/e2 NULL. Mirrors build_flat_catalog_struct with metric
    METRIC_3DBOX (=2).

    Returns
    -------
    (MultiresoCatalog, list)
    """
    p1 = np.ascontiguousarray(sh['pos1'], dtype=np.float64)
    p2 = np.ascontiguousarray(sh['pos2'], dtype=np.float64)
    p3 = np.ascontiguousarray(sh['pos3'], dtype=np.float64)
    w  = np.ascontiguousarray(sh['weight'], dtype=np.float64)
    zb = np.ascontiguousarray(sh['zbins'], dtype=np.int32)
    ii = np.ones(len(p1), dtype=np.float64)   # box has no buffer galaxies
    ngal = np.array([len(p1)], dtype=np.int32)
    keepers = [p1, p2, p3, w, zb, ii, ngal]

    s = MultiresoCatalog()
    s.metric        = 2                        # METRIC_3DBOX
    s.nresos        = 1
    s.nbinsz        = int(nbinsz)
    s.ngal_resos    = _ptr_i32(ngal)
    s.isinner_resos = _ptr_f64(ii)
    s.weight_resos  = _ptr_f64(w)
    s.zbin_resos    = _ptr_i32(zb)
    s.pos1_resos    = _ptr_f64(p1)
    s.pos2_resos    = _ptr_f64(p2)
    s.pos3_resos    = _ptr_f64(p3)
    if e1e2 is not None:
        e1 = np.ascontiguousarray(e1e2[0], dtype=np.float64)
        e2 = np.ascontiguousarray(e1e2[1], dtype=np.float64)
        keepers += [e1, e2]
        s.e1_resos = _ptr_f64(e1)
        s.e2_resos = _ptr_f64(e2)

    return s, keepers


def build_slab_navhash_struct(sh):
    """'3dbox' NavHash from a slab hash (the per-leg transverse+LOS navigation).

    The transverse plane uses the shared per-slab 2D hash (``index_matcher`` is
    length ``nslabs*npix``, indexed ``s*npix + ...``; ``pixs_galind_bounds`` is
    the concatenated per-slab bounds, offset by ``rshift_bounds[s]``; ``pix_gals``
    holds the reordered global galaxy index, offset by ``slab_offsets[s]``). Only
    leg catalogs (hashed) need a NavHash; the looped central carries none.

    Returns
    -------
    (NavHash, list)
    """
    im  = np.ascontiguousarray(sh['index_matcher'],      dtype=np.int32)
    pgb = np.ascontiguousarray(sh['pixs_galind_bounds'], dtype=np.int32)
    pg  = np.ascontiguousarray(sh['pix_gals'],           dtype=np.int32)
    so  = np.ascontiguousarray(sh['slab_offsets'],       dtype=np.int32)
    rsb = np.ascontiguousarray(sh['rshift_bounds'],      dtype=np.int32)
    keepers = [im, pgb, pg, so, rsb]

    s = NavHash()
    s.metric             = 2                   # METRIC_3DBOX
    s.index_matcher      = _ptr_i32(im)
    s.pixs_galind_bounds = _ptr_i32(pgb)
    s.pix_gals           = _ptr_i32(pg)
    s.pix1_start = float(sh['pix1_start'])
    s.pix1_d     = float(sh['pix1_d'])
    s.pix1_n     = int(sh['pix1_n'])
    s.pix2_start = float(sh['pix2_start'])
    s.pix2_d     = float(sh['pix2_d'])
    s.pix2_n     = int(sh['pix2_n'])
    s.slab_offsets  = _ptr_i32(so)
    s.rshift_bounds = _ptr_i32(rsb)
    s.nslabs = int(sh['nslabs'])
    s.z0     = float(sh['z0'])
    s.dpix_z = float(sh['dpix_z'])

    return s, keepers


def build_tree_params_struct(corr, mh):
    """Populate a TreeResoParams from a correlator instance and a bundle.

    ``reso_redges`` must be expressed in the same unit as ``BinningParams.rmin``
    (radians on the spherical path, ``sep_units`` on the flat path). The
    spherical bundle already returns radian band edges under ``mh['reso_redges']``;
    the flat path uses the correlator's ``tree_redges`` (in ``sep_units``, the
    same unit as the flat positions).

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
        re  = np.ascontiguousarray(mh['reso_redges'], dtype=np.float64)  # radians
    keepers += [dp1, dp2, re]

    s = TreeResoParams()
    s.nresos          = int(corr.tree_nresos)
    s.nresos_grid     = int(corr.tree_nresos)   # NN ignores this; GGG caller adjusts
    s.dpix1_resos     = _ptr_f64(dp1)
    s.dpix2_resos     = _ptr_f64(dp2)
    s.reso_redges     = _ptr_f64(re)
    s.resoshift_leafs = int(corr.resoshift_leafs)
    s.minresoind_leaf = int(corr.minresoind_leaf)
    s.maxresoind_leaf = int(corr.maxresoind_leaf)

    return s, keepers


def build_binning_struct(corr, scale=None, do_dc=None, nmax=0, nmin=0, dccorr=0,
                         Pi=0., rbins=None):
    """Populate a BinningParams from a correlator instance.

    Parameters
    ----------
    corr : BinnedNPCF subclass
    scale : float or None
        Factor converting ``min_sep``/``max_sep`` from the correlator's
        ``sep_units`` into the working unit passed to C (radians on the spherical
        path). ``None`` leaves the values in ``sep_units`` (flat path).
    nmax, nmin : int
        Multipole order range (GGG and higher); ignored by NN/GG.
    dccorr : int
        Multi-count correction toggle (GGG); ignored by NN/GG.
    Pi : float
        Line-of-sight window half-width (METRIC_3DBOX); ignored otherwise.
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
    s.rmin   = float(corr.min_sep) * f
    s.rmax   = float(corr.max_sep) * f
    s.nbinsr = int(corr.nbinsr)
    s.do_dc  = int(1) if do_dc is None else int(do_dc)
    s.nmax   = int(nmax)
    s.nmin   = int(nmin)
    s.dccorr = int(dccorr)
    s.Pi     = float(Pi)
    s.rbins  = _null(_p_f64)
    if rbins is not None:
        rb = np.ascontiguousarray(rbins, dtype=np.float64)
        s._rbins_keep = rb
        s.rbins = _ptr_f64(rb)
    return s


def build_npcf_output(kind, nbinsr, nmax=0, nbinsz=None, nbinsz_lens=None,
                      nbinsz_source=None, bc_len=None,
                      npcf_len=None, norm_mp_len=None, ncomp=None):
    """Allocate zeroed output arrays and wrap them in a unified NPCFOutput.

    Single builder for every order (see NPCFOutput in multires_structs.h). Only
    the arrays live for ``kind`` are allocated; the rest of the returned tuple is
    ``None`` and the matching struct fields stay NULL.

    Parameters
    ----------
    kind : {'nn', 'gg', 'ng', 'ggg', 'gnn'}
        Correlator family. 'gnn' also serves NGG (same Upsilon_n/Norm_n layout).
    nbinsr : int
    nmax : int
        Multipole order (ggg/gnn); 0 for nn/gg/ng.
    nbinsz, nbinsz_lens, nbinsz_source : int
        Tomographic bin counts (per kind: nn/gg/ggg use ``nbinsz``; ng uses
        lens+source).
    bc_len, npcf_len, norm_mp_len, ncomp : int
        For 'gnn'/'ngg' -- the flat lengths of bin_centers / npcf (Upsilon_n) /
        norm_mp (Norm_n) and the natural-component count, all layout-specific and
        passed explicitly by the caller (from its reshape shapes).

    Returns
    -------
    (NPCFOutput, bin_centers, npcf, norm, norm_mp, npair, npair_cell)
        Unused arrays are ``None``.
    """
    s = NPCFOutput()
    s.nmax = int(nmax)
    bin_centers = npcf = norm = norm_mp = npair = npair_cell = None

    if kind == 'nn':
        z2r = nbinsz*nbinsz*nbinsr
        bin_centers = np.zeros(z2r, dtype=np.float64)
        norm        = np.zeros(z2r, dtype=np.float64)   # weighted pair count
        npair_cell  = np.zeros(z2r, dtype=np.int64)
        s.norm       = _ptr_f64(norm)
        s.npair_cell = _ptr_i64(npair_cell)
        s.ncomp = 0
    elif kind == 'gg':
        z2r = nbinsz*nbinsz*nbinsr
        bin_centers = np.zeros(z2r, dtype=np.float64)
        npcf        = np.zeros(2*z2r, dtype=np.complex128)   # [xip, xim] stacked
        norm        = np.zeros(z2r, dtype=np.float64)
        npair       = np.zeros(z2r, dtype=np.int64)
        s.npcf  = npcf.ctypes.data_as(_p_c128)
        s.norm  = _ptr_f64(norm)
        s.npair = _ptr_i64(npair)
        s.ncomp = 2
    elif kind == 'ng':
        z2r = nbinsz_lens*nbinsz_source*nbinsr
        bin_centers = np.zeros(z2r, dtype=np.float64)
        npcf        = np.zeros(z2r, dtype=np.complex128)
        norm        = np.zeros(z2r, dtype=np.float64)
        npair       = np.zeros(z2r, dtype=np.int64)
        s.npcf  = npcf.ctypes.data_as(_p_c128)
        s.norm  = _ptr_f64(norm)
        s.npair = _ptr_i64(npair)
        s.ncomp = 1
    elif kind == 'ggg':
        nzc = nbinsz*nbinsz*nbinsz
        comp = (nmax+1)*nzc*nbinsr*nbinsr
        bin_centers = np.zeros(nbinsz*nbinsr, dtype=np.float64)
        npcf        = np.zeros(4*comp, dtype=np.complex128)   # Gammans (4 comps)
        norm_mp     = np.zeros(comp, dtype=np.complex128)     # Gammans_norm
        s.npcf    = npcf.ctypes.data_as(_p_c128)
        s.norm_mp = norm_mp.ctypes.data_as(_p_c128)
        s.ncomp = 4
    elif kind in ('gnn', 'ngg'):
        # G3L / IA multipole outputs (Upsilon_n, Norm_n). Layout varies (GNN:
        # ncomp=1, nmax+1 multipoles; NGG: ncomp=2, 2*nmax+1), so the caller
        # passes the flat lengths it already knows from its reshape shapes.
        bin_centers = np.zeros(bc_len, dtype=np.float64)
        npcf        = np.zeros(npcf_len, dtype=np.complex128)     # Upsilon_n
        norm_mp     = np.zeros(norm_mp_len, dtype=np.complex128)  # Norm_n
        s.npcf    = npcf.ctypes.data_as(_p_c128)
        s.norm_mp = norm_mp.ctypes.data_as(_p_c128)
        s.ncomp = int(ncomp)
    else:
        raise ValueError(f"unknown NPCFOutput kind {kind!r}")

    s.bin_centers = _ptr_f64(bin_centers)
    return s, bin_centers, npcf, norm, norm_mp, npair, npair_cell
