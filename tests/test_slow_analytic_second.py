# Here we collect slow-tier tests that make sure the third-order correlator performs the
# correct measurement by comparing against a mock with an analytically known 2pcf

import numpy as np

from conftest import NGAL_SECOND, R0_SLOW, RTOL_LOOSE, RTOL_TIGHT, assert_amp_phase

 # Amplitude floor per sect 8.2 in the notes: xi_pm each change sign twice inside this range
THEORY_FLOOR = .05

# Make sure that the predicted window effect is correct. This is required for
# the correction to the measurements to work.
def test_pair_counts_reproduce_the_edge_window(field, nn_measured):
    _, nn = nn_measured
    edges = nn.bin_edges
    cen = np.sqrt(edges[1:]*edges[:-1])
    ring = np.pi*(edges[1:]**2 - edges[:-1]**2)
    expect = NGAL_SECOND**2/field.area*ring*field.f_pair(cen)
    assert np.max(np.abs(np.asarray(nn.npair).ravel()/expect - 1.)) < RTOL_TIGHT

# Do the test for xipm
# We need 250k gals for decent convergence in tight tier. Note that the discrepancy
# here is dominated by the sampling floor and not by the estimator.
def test_xi_pm_reproduce_the_closed_form(field, gg_measured):
    gg, (th_p, th_m, win) = gg_measured
    xip = np.asarray(gg.xip).ravel().real*win
    xim = np.asarray(gg.xim).ravel()*win
    cen = np.asarray(gg.bin_centers_mean).ravel()
    sel_base = cen > field.min_usable_sep(NGAL_SECOND)
    assert sel_base.sum() > 20, "too few bins above the grid-step floor to be meaningful"
    for meas, theory, name in ((xip, th_p, 'xi_plus'), (xim, th_m, 'xi_minus')):
        sel_theo = np.abs(theory) > THEORY_FLOOR*np.max(np.abs(theory[sel_base]))
        keep = sel_base & sel_theo
        assert keep.sum() > 15, (name, "too few bins above the theory floor")
        assert_amp_phase(meas[keep], theory[keep], RTOL_TIGHT, RTOL_TIGHT, name)

# Do the test for nn
# We need 250k gals for decent convergence in loose tier. Note that the discrepancy
# here is dominated by the shot noise and not by the estimator.
def test_omega_reproduces_the_closed_form(field, nn_measured):
    nn_d, nn_r = nn_measured
    cen = np.asarray(nn_d.bin_centers_mean).ravel()
    count = np.asarray(nn_r.npair).ravel()
    meas = (np.asarray(nn_d.npair).ravel()/count - 1.)*field.f_pair(cen)
    theory = field.omega(cen)
    # xi_plus (and so omega) crosses zero at 1.531 r0; stay inside the first, dominant lobe
    keep = (cen > field.min_usable_sep(NGAL_SECOND)) & (cen < 1.3*R0_SLOW) & (count > 100)
    keep &= np.abs(theory) > THEORY_FLOOR*np.max(np.abs(theory[keep]))
    assert keep.sum() > 8, "too few bins in the well-resolved range to be meaningful"
    assert np.allclose(meas[keep], theory[keep], rtol=RTOL_LOOSE, atol=0.)
