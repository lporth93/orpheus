# Here we test some easy conventions regarding overall signs. This is done
# against a field with theoretically known gamma_t,x.
# As this is a fast tier test we do not care about convergence of the estimators.

import numpy as np
import pytest

from orpheus.catalog import SpinTracerCatalog
from orpheus.npcf_second import GGCorrelation, NGCorrelation
from orpheus.sphericalmap import SphericalMap

from conftest import (CORRELATORS, MAX_SEP, MIN_SEP, NBINSR, NTHREADS, RTOL_EXACT,
                      TREE_ONLY, build_correlator, correlator_ids, correlator_outputs,
                      correlators, run_correlator)

TREE_RESOS = [0., 2., 4.]
RMIN_PIXSIZE = 16
THIRD_MAX_SEP = 20.
THIRD_NMAX = 6
THIRD_NBINSPHI = 30
THISAPR = np.geomspace(2., 15., 5)


def _ng(cat_source, cat_lens):
    ng = NGCorrelation(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR,
                       tree_resos=TREE_RESOS, rmin_pixsize=RMIN_PIXSIZE,
                       shuffle_pix=1, nthreads=NTHREADS)
    ng.process(cat_source, cat_lens, dotomo=False)
    return ng

# Check if estimator for NG & MapNap produces correct sign for gamma_t>0 field
def test_ng_returns_positive_tangential_shear(tangential_field):

    # The input field has gamma=gamma_t+i*0, so we assert this
    cat_source, cat_lens, gamma_t = tangential_field

    ng = _ng(cat_source, cat_lens)
    mapnap = ng.computeMapNap(THISAPR)
    assert np.max(np.abs(ng.xi[0].real - gamma_t)) < 1e-3
    assert np.max(np.abs(ng.xi[0].imag)) < 1e-3
    assert np.all(mapnap[0].real > 0.)
    assert np.all(np.abs(mapnap[0].imag) < 1e-3)

# Check whether E-Mode field becomes B-Mode field when rotating shear by pi/4
def test_ng_cross_shear_flips_into_the_imaginary_part(tangential_field):
    cat_source, cat_lens, gamma_t = tangential_field
    tracer_rot = 1j*(cat_source.tracer_1 + 1j*cat_source.tracer_2)
    rotated = SpinTracerCatalog(spin=2, pos1=cat_source.pos1, pos2=cat_source.pos2,
                                tracer_1=tracer_rot.real, tracer_2=tracer_rot.imag,
                                weight=cat_source.weight, geometry='flat2d')
    ng = _ng(rotated, cat_lens)
    assert np.max(np.abs(ng.xi[0].real)) < 1e-3
    assert np.max(np.abs(ng.xi[0].imag - gamma_t)) < 1e-3


def test_map2_fourth_component_is_minus_the_second(quadrupole_field):
    """computeMap2 returns [Map2, MapMx, Mx2, -MapMx]; the last is not independent."""
    # Note that here we need to enforce do_dc=True as only including double-counting
    # fulfils the algebraic condition one of the xi-components being exactly zero. This
    # then translates to the Map2-basis as MapMx=-MxMap
    cat, _ = quadrupole_field
    gg = GGCorrelation(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR,
                       tree_resos=[0., 2.], rmin_pixsize=RMIN_PIXSIZE,
                       shuffle_pix=1, nthreads=NTHREADS)
    gg.process(cat, dotomo=False, do_dc=True)
    m2 = gg.computeMap2(THISAPR)
    assert m2.shape[0] == 4
    assert np.max(np.abs(m2[1] + m2[3])) < RTOL_EXACT*np.max(np.abs(m2[0]))

# For parity tests only select the relevant correlators, i.e. the ones having at lease on spin2 leg
PARITY = correlators(spin2=True)

# Define all the kwargs for all correlators
def _parity_kwargs(spec):
    if spec.order == 2:
        return dict(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR,
                    tree_resos=[0., 2.], rmin_pixsize=RMIN_PIXSIZE, shuffle_pix=1,
                    nthreads=NTHREADS)
    kwargs = dict(min_sep=MIN_SEP, max_sep=THIRD_MAX_SEP, nbinsr=NBINSR,
                  nmaxs=THIRD_NMAX, nbinsphi=THIRD_NBINSPHI, nthreads=NTHREADS)
    if spec.cls.__name__ in TREE_ONLY:
        return dict(kwargs, method=TREE_ONLY[spec.cls.__name__],
                    tree_resos=[0., 2., 4.], rmin_pixsize=8)
    return dict(kwargs, method='Discrete')


@pytest.mark.parametrize("spec", PARITY, ids=correlator_ids(PARITY))
def test_parity_under_a_global_shear_flip(spec, quadrupole_field, scalar_catalog):
    """Flipping every shear multiplies the correlator by (-1)^nspin2."""
    cat, _ = quadrupole_field
    flipped = SpinTracerCatalog(spin=2, pos1=cat.pos1, pos2=cat.pos2,
                                tracer_1=-cat.tracer_1, tracer_2=-cat.tracer_2,
                                weight=cat.weight, geometry='flat2d')
    out = []
    for shear in (cat, flipped):
        inst = build_correlator(spec, **_parity_kwargs(spec))
        run_correlator(spec, inst, shear, scalar_catalog,
                       **(dict(statistics='4pcf_multipole') if spec.order == 4 else {}))
        out.append([np.asarray(getattr(inst, f)) for f in correlator_outputs(spec)])
    for name, ref, flip in zip(correlator_outputs(spec), *out):
        assert np.allclose(flip, (-1)**spec.nspin2*ref, rtol=RTOL_EXACT, atol=0.), name


# Check against the orpheus package that the number of spin-2 legs and the order are correctly
# put in to the CORRELATORS table in conftest
def test_spins_match_the_number_of_polar_legs():
    for spec in CORRELATORS:
        inst = build_correlator(spec, min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR)
        assert int(np.sum(np.asarray(inst.spins) == 2)) == spec.nspin2, spec.cls.__name__
        assert len(np.asarray(inst.spins)) == spec.order, spec.cls.__name__

# Make sure that aperture mass map of pure E-Mode field does not leak a B-Mode at high declination
@pytest.mark.parametrize("dec0", [15., 80.])
def test_sphericalmap_tangential_field_carries_no_cross_mode(dec0):
    from healpy import ang2pix, pix2vec
    nside, R_ap, gamma_t, ngal = 256, 20., .1, 4000
    rng = np.random.default_rng(9)

    # Take the center to be a pixel center, so the estimator evaluates exactly there
    cpix = ang2pix(nside, np.radians(90.-dec0), np.radians(40.), nest=True)
    cvec = np.asarray(pix2vec(nside, cpix, nest=True))
    east = np.array([-cvec[1], cvec[0], 0.]); east /= np.linalg.norm(east)
    north = np.cross(cvec, east)

    # Tracers spread over the filter support around that center
    theta = 4.*R_ap*np.pi/(180.*60.)*np.sqrt(rng.uniform(.01, 1., ngal))
    phi = rng.uniform(0., 2.*np.pi, ngal)
    gvec = (cvec[:, None]*np.cos(theta)
            + np.sin(theta)*(np.cos(phi)*east[:, None] + np.sin(phi)*north[:, None]))
    gvec /= np.linalg.norm(gvec, axis=0)
    ra = np.degrees(np.arctan2(gvec[1], gvec[0]))%360.
    dec = np.degrees(np.arcsin(np.clip(gvec[2], -1., 1.)))

    # Map gamma_t to sphere. (Note that in some sense this just checks that we use the 
    # same bearing convention in C...but it was also verified on "real" data.)
    dsq = np.sum((gvec - cvec[:, None])**2, axis=0)
    bear = np.arctan2((cvec[2] - gvec[2]) + .5*gvec[2]*dsq,
                      gvec[0]*cvec[1] - gvec[1]*cvec[0])
    ell = -gamma_t*np.exp(2j*bear)
    cat = SpinTracerCatalog(spin=2, pos1=ra, pos2=dec, tracer_1=ell.real, tracer_2=ell.imag,
                            weight=np.ones(ngal), geometry='spherical',
                            units_pos1='deg', units_pos2='deg')

    smap = SphericalMap(nside=nside, R_ap=R_ap, sep_units='arcmin', method="Discrete",
                        nthreads=NTHREADS)
    smap.process(cat, dotomo=False, centers=np.array([cpix]), approx_coverage=False)
    got = smap.Map[0][cpix]
    assert got.real > 0., "a purely tangential field has to give positive Map"
    assert abs(got.imag) < RTOL_EXACT*abs(got.real)
