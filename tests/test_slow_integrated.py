# Here we compare the aperture statistics obtained form integrating the NPCFs in orpheus
# to the theoretical expressions
# --> Sensitive to convergence of the actual integral and to accuracy/convergence of the npcf 
#     estimator. See related script test_slow_integrated_consistency.py for test using theo npcfs
# Recall that here we are working with a rotated shear field, so we expect significant measurement
# for all different components of Mapn

import numpy as np

from orpheus.npcf_second import NGCorrelation

from conftest import (NGAL_SECOND, NTHREADS_SLOW, R0_SLOW, RTOL_LOOSE, RTOL_NORMAL,
                      RTOL_TIGHT, TREE_SLOW, kept_radii, RECOMMENDED, NCROSS)

R0 = R0_SLOW
APR = np.geomspace(*RECOMMENDED['aperture_radii'], 6)*R0
AP_FLOOR = .1 # Only consider statistics which are not too far off the peak


# Do the test for map2
def test_map2_ebmodes_reproduce_the_closed_form(field, gg_measured):
    gg, (_, _, win) = gg_measured
    # Get map2 from measured xipm, corrected by finite-field window
    xip, xim = np.asarray(gg.xip).copy(), np.asarray(gg.xim).copy()
    gg.xip, gg.xim = xip*win, xim*win
    meas = np.asarray(gg.computeMap2(APR))[:, 0]
    gg.xip, gg.xim = xip, xim
    # Compare against theo
    theo = field.map_n_ebmodes(2, APR)
    for j in range(3):
        k = kept_radii(theo[j], .05)
        assert np.allclose(meas[j][k], theo[j][k], rtol=RTOL_NORMAL, atol=0.), j

# Do the test for mapnap
def test_nap_map_reproduces_the_closed_form(field):

    # Measure NG for both signs of delta and rescale the resulting xi with the window
    xis = {}
    for sign in (1., -1.):
        cat, lens = field.catalogs(NGAL_SECOND, delta_sign=sign)
        ng = NGCorrelation(min_sep=RECOMMENDED['min_sep_second']*R0,
                           max_sep=RECOMMENDED['max_sep']*R0, binsize=.1,
                           nthreads=NTHREADS_SLOW, **TREE_SLOW)
        ng.process(cat, lens, dotomo=False)
        xis[sign] = np.asarray(ng.xi).copy()
    cen = np.asarray(ng.bin_centers_mean).ravel()
    ng.xi = field.parity_combine(xis[1.], xis[-1.], 1)*field.f_pair(cen)
    meas = np.asarray(ng.computeMapNap(APR)).ravel()
    # Compare against theory for converged radii
    for part, ncross in ((meas.real, 0), (meas.imag, 1)):
        theo = field.nap_map_n(1, 1, APR, ncross=ncross)
        k = kept_radii(theo, AP_FLOOR)
        assert np.allclose(part[k], theo[k], rtol=RTOL_TIGHT, atol=0.), ncross

# Do the test for nap2
# Note that here, both the integral and the estimator converge very slow, so only loose tier
def test_nap2_reproduces_the_closed_form(field, nn_measured):
    nn_d, nn_r = nn_measured
    # Get nap2 from measured xipm, corrected by finite-field window
    cen = np.asarray(nn_d.bin_centers_mean).ravel()
    meas = np.asarray(nn_d.npair).ravel()/np.asarray(nn_r.npair).ravel() - 1.
    xi_orig = nn_d.xi
    nn_d.xi = (meas*field.f_pair(cen))[None, :]
    apr = np.geomspace(.95, 1.05, 3)*R0
    meas = nn_d.computeNap2(apr)[0]
    nn_d.xi = xi_orig
    theo = field.nap_map_n(2, 0, apr)
     # Compare against theory
    assert np.allclose(meas, theo, rtol=RTOL_LOOSE, atol=0.), meas/theo

# Do the test for map3
def test_map3_ebmodes_reproduce_the_closed_form(field, ggg_measured):
    ggg, _, window = ggg_measured
    # Get map3 from measured npcf, corrected by finite-field window
    npcf = np.asarray(ggg.npcf).copy()
    ggg.npcf = npcf*window[None, None]
    meas = np.asarray(ggg.computeMap3(APR, basis='MapMx'))[:, 0]
    ggg.npcf = npcf
    # Compare against theo
    theo = field.map_n_ebmodes(3, APR)[NCROSS[3]]
    # measured 1.5e-2 worst over the kept radii (67% of them), i.e. NORMAL
    for j in range(8):
        k = kept_radii(theo[j], .05)
        assert np.allclose(meas[j][k], theo[j][k], rtol=RTOL_NORMAL, atol=0.), j

# Aside: MapMxMap etc have to be the same for equal-scale stats in theory. Due 
# to the binning scheme this is does not hold exactly for the estimator geometry
# (inferring r3 from r1,r2 bins) therefore estimate of integral/binning convergence
def test_map3_permutations_agree_at_equal_aperture_radii(ggg_measured):
    ggg, _, _ = ggg_measured
    out = np.asarray(ggg.computeMap3(APR, basis='MapMx'))[:, 0]
    scale = np.max(np.abs(out[0]))
    for group in ([1, 2, 3], [4, 5, 6]):
        spread = np.max(np.abs(out[group] - out[group].mean(axis=0)))/scale
        assert spread < RTOL_TIGHT, (group, spread)

# Do the test for map3 (unequal)
def test_map3_reproduces_the_closed_form_at_unequal_aperture_scales(field, ggg_measured):

    ggg, theory_eq, window = ggg_measured
    # Get map3 from measured npcf, corrected by finite-field window
    npcf_orig = np.asarray(ggg.npcf).copy()
    ggg.npcf = npcf_orig*window[None, None]
    radii = np.geomspace(*RECOMMENDED['aperture_radii'], 3)*R0
    map3 = np.asarray(ggg.computeMap3(radii, do_multiscale=True, basis='MapMx'))[:, 0]
    ggg.npcf = npcf_orig
    cos3 = np.cos(field.chi)**3
    meas, theo = [], []
    idx = 0
    for R1 in radii:
        for R2 in radii:
            for R3 in radii:
                theo.append(np.ravel(field.map_unequal([R1, R2, R3])*cos3)[0])
                meas.append(map3[0, idx])
                idx += 1
    # Compare against theo
    meas, theo = np.array(meas), np.array(theo)
    k = kept_radii(theo, .05)
    assert np.allclose(meas[k], theo[k], rtol=RTOL_NORMAL, atol=0.), meas[k]/theo[k]

# Do the test for mapnap2
def test_nnm_reproduces_the_closed_form(field, gnn_measured):
    gnn, combined, _, window = gnn_measured
    # Get mapnap2 from measured npcf, corrected by finite-field window
    npcf_orig = np.asarray(gnn.npcf).copy()
    gnn.npcf = (combined*window)[None, None]
    meas = np.asarray(gnn.computeNNM(APR))[0, 0]
    # Compare against theo
    gnn.npcf = npcf_orig
    for part, ncross in ((meas.real, 0), (meas.imag, 1)):
        theo = field.nap_map_n(2, 1, APR, ncross=ncross)
        k = kept_radii(theo, AP_FLOOR)
        assert np.allclose(part[k], theo[k], rtol=RTOL_NORMAL, atol=0.), ncross

# Do the test for map2nap
def test_nmm_reproduces_the_closed_form(field, ngg_measured):
    ngg, combined, _, window = ngg_measured
    # Get map2nap from measured npcf, corrected by finite-field window
    npcf_orig = np.asarray(ngg.npcf).copy()
    ngg.npcf = (combined*window[None])[:, None]
    meas = np.asarray(ngg.computeNMM(APR, basis='MapMx'))[:, 0]
    ngg.npcf = npcf_orig
    # Compare against theo
    for i, ncross in enumerate((0, 2, 1, 1)):
        theo = field.nap_map_n(1, 2, APR, ncross=ncross)
        k = kept_radii(theo, AP_FLOOR)
        assert np.allclose(meas[i][k], theo[k], rtol=RTOL_TIGHT, atol=0.), i