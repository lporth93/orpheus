# Here we compare the aperture statistics obtained form integrating the NPCFs in orpheus
# analogous computation from theory, i.e. integrating the bin-averaged theory npcfs using 
# the same binning scheme used for the measurement 
# --> Insensitive to convergence of the actual integral and solely to the npcf estimator 
#     accuracy/convergence
# Nap2 is not included as the measured 2pcf remains so noisy that it barely makes loose tier

import ctypes as ct

import numpy as np
from orpheus.npcf_fourth import GGGGCorrelation_NoTomo

from conftest import RTOL_EXACT, RTOL_SHARP, RTOL_TIGHT, kept_radii, RECOMMENDED

R0 = 3.
APR = np.geomspace(*RECOMMENDED['aperture_radii'], 6)*R0
# GGGG spans only 0.6-1.6 r0 in separation, so three radii already sample the usable
# aperture range; the truncation this implies cancels between the two sides anyway.
APR4 = np.geomspace(*RECOMMENDED['aperture_radii'], 3)*R0
AP_FLOOR = .1


# Little helper to get map4 from the 4pcf in the real-space basis. 
# At the moment not included in the main package (there only for multipole-space basis)
# so we extend this here
def _m4_from_npcf(gggg, npcf, radii):

    # Link to the instance clib and wrap the relevant function
    clib = gggg.clib
    p_f64 = np.ctypeslib.ndpointer(dtype=np.float64)
    p_c128 = np.ctypeslib.ndpointer(dtype=np.complex128)
    clib.fourpcf2M4correlators.restype = ct.c_void_p
    clib.fourpcf2M4correlators.argtypes = [
        ct.c_int32,
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        p_f64, p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32, p_c128, p_c128]

    # For each radial bin, update the M4 integral value
    edges = np.ascontiguousarray(gggg.bin_edges, dtype=np.float64)
    cen = np.ascontiguousarray(gggg.bin_centers_mean, dtype=np.float64)
    p1, p2 = (np.ascontiguousarray(p, dtype=np.float64) for p in gggg.phis[:2])
    d1, d2 = (np.ascontiguousarray(d, dtype=np.float64) for d in gggg.dphis[:2])
    m4 = np.zeros((8, len(radii)), dtype=np.complex128)
    buf = np.zeros(8, dtype=np.complex128)
    for i in range(len(cen)):
        for j in range(len(cen)):
            for k in range(len(cen)):
                sl = np.ascontiguousarray(
                    npcf[:, i, j, k].reshape(8, -1), dtype=np.complex128).ravel()
                for ir, R in enumerate(radii):
                    buf[:] = 0.
                    clib.fourpcf2M4correlators(
                        1, cen[i]/R, cen[j]/R, cen[k]/R,
                        (edges[i+1]-edges[i])/R, (edges[j+1]-edges[j])/R,
                        (edges[k+1]-edges[k])/R,
                        p1, p2, d1, d2, len(p1), len(p2), sl, buf)
                    # the kernel leaves NaN where a configuration cannot contribute
                    good = ~np.isnan(np.abs(buf))
                    m4[good, ir] += buf[good]
    return m4


# Do the test for map2
def test_map2_matches_the_transform_of_the_binned_theory(gg_measured):
    gg, (txip, txim, win) = gg_measured
    # Get map2 from measured xipm, corrected by finite-field window
    xip, xim = np.asarray(gg.xip).copy(), np.asarray(gg.xim).copy()
    gg.xip, gg.xim = xip*win, xim*win
    meas = np.asarray(gg.computeMap2(APR))[:, 0]
    # Get map2 from theory xipm
    gg.xip = np.broadcast_to(txip, xip.shape).copy()
    gg.xim = np.broadcast_to(txim, xim.shape).copy()
    theo = np.asarray(gg.computeMap2(APR))[:, 0]
    gg.xip, gg.xim = xip, xim
    for j in range(3):
        k = kept_radii(theo[j], AP_FLOOR)
        assert np.allclose(meas[j][k], theo[j][k], rtol=RTOL_SHARP, atol=0.), j

# Do the test for map3 (equal)
def test_map3_matches_the_transform_of_the_binned_theory(ggg_measured):
    ggg, theory, window = ggg_measured
    # Get map3 from measured npcf, corrected by finite-field window
    npcf = np.asarray(ggg.npcf).copy()
    ggg.npcf = npcf*window[None, None]
    meas = np.asarray(ggg.computeMap3(APR, basis='MapMx'))[:, 0]
    # Get map3 from theory npcf
    ggg.npcf = np.broadcast_to(theory[:, None], npcf.shape).copy()
    theo = np.asarray(ggg.computeMap3(APR, basis='MapMx'))[:, 0]
    ggg.npcf = npcf
    for j in range(8):
        k = kept_radii(theo[j], AP_FLOOR)
        assert np.allclose(meas[j][k], theo[j][k], rtol=RTOL_SHARP, atol=0.), j

# Do the test for map3 (unequal)
def test_map3_unequal_matches_the_transform_of_the_binned_theory(ggg_measured):
    ggg, theory, window = ggg_measured
    # Get map3 from measured npcf, corrected by finite-field window
    npcf = np.asarray(ggg.npcf).copy()
    ggg.npcf = npcf*window[None, None]
    meas = np.asarray(ggg.computeMap3(APR4, do_multiscale=True, basis='MapMx'))[0, 0]
    # Get map3 from theory npcf
    ggg.npcf = np.broadcast_to(theory[:, None], npcf.shape).copy()
    theo = np.asarray(ggg.computeMap3(APR4, do_multiscale=True, basis='MapMx'))[0, 0]
    ggg.npcf = npcf
    k = kept_radii(theo, AP_FLOOR)
    assert np.allclose(meas[k], theo[k], rtol=RTOL_SHARP, atol=0.), meas[k]/theo[k]

# Do the test for mapnap2
# As GNN itself is noisy even this test only makes it to tight tier
def test_nnm_matches_the_transform_of_the_binned_theory(gnn_measured):
    gnn, combined, theory, window = gnn_measured
    # Get mapnap2 from measured npcf, corrected by finite-field window
    orig = np.asarray(gnn.npcf).copy()
    gnn.npcf = (combined*window)[None, None]
    meas = np.asarray(gnn.computeNNM(APR))[0, 0]
    # Get mapnap2 from theory npcf
    gnn.npcf = theory[None, None]
    theo = np.asarray(gnn.computeNNM(APR))[0, 0]
    gnn.npcf = orig
    for part, tag in ((np.real, 'NapNapMap'), (np.imag, 'NapNapMx')):
        k = kept_radii(part(theo), AP_FLOOR)
        assert np.allclose(part(meas)[k], part(theo)[k], rtol=RTOL_TIGHT, atol=0.), tag

# Do the test for map2nap
def test_nmm_matches_the_transform_of_the_binned_theory(ngg_measured):
    ngg, combined, theory, window = ngg_measured
    # Get map2nap from measured npcf, corrected by finite-field window
    orig = np.asarray(ngg.npcf).copy()
    ngg.npcf = (combined*window[None])[:, None]
    meas = np.asarray(ngg.computeNMM(APR, basis='MapMx'))[:, 0]
    # Get map2nap from theory npcf
    ngg.npcf = theory[:, None]
    theo = np.asarray(ngg.computeNMM(APR, basis='MapMx'))[:, 0]
    ngg.npcf = orig
    for i in range(4):
        k = kept_radii(theo[i], AP_FLOOR)
        assert np.allclose(meas[i][k], theo[i][k], rtol=RTOL_SHARP, atol=0.), i

# Do the test for map4
def test_map4_matches_the_transform_of_the_binned_theory(gggg_measured):
    """2.3e-3 over the 16 MapMx components -- the only quantitative check on Map4 in the
    suite, since ``test_slow_analytic_fourth.py`` stops at the natural components and
    there is no Map4 closed form in ``reference.py``.

    The first assertion pins the python driver against ``computeMap4`` itself (agreement
    is at 1e-15), without which the second would only be comparing the driver to itself.

    Asserted on every radius rather than over an amplitude cut: the error here sits at
    2-4e-3 across six decades of |ref| (3.7e-11 down to 9.8e-17), so the cut removes no
    noise and would leave a single radius out of three. TIGHT rather than SHARP because
    the uncut worst is 4.3e-3, which would sit on SHARP's edge.
    """
    gggg, theory, window = gggg_measured
    npcf = np.asarray(gggg.npcf)[:, 0]
    native = np.asarray(gggg.computeMap4(APR4, basis='MM*'))[0]
    driven = _m4_from_npcf(gggg, npcf, APR4)
    # Just a check that our helper _m4_from_npcf works
    assert np.allclose(driven, native, rtol=RTOL_EXACT, atol=0.), "driver != computeMap4"

    meas = GGGGCorrelation_NoTomo.MMStar2MapMx_fourth(_m4_from_npcf(gggg, npcf*window[None], APR4))
    theo = GGGGCorrelation_NoTomo.MMStar2MapMx_fourth(_m4_from_npcf(gggg, theory, APR4))
    meas, theo = np.asarray(meas), np.asarray(theo)
    for i in range(meas.shape[0]):
        assert np.allclose(meas[i], theo[i], rtol=RTOL_TIGHT, atol=0.), i
