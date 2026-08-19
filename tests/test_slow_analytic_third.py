# Here we collect slow-tier tests that make sure the third-order correlator performs the
# correct measurement by comparing against a mock with an analytically known 3pcf

import numpy as np

from conftest import (MIN_KEPT, RTOL_NORMAL, RTOL_TIGHT, assert_amp_phase,
                      kept_configurations)

# Per-component amplitude floors, notes sect 8.2
THEORY_FLOORS = (.15, .05, .10, .10)
# Dont consider bins which might be rather empty
NORM_PCT = 75

def _kept(ggg, theory, window, floor):
    norm = np.asarray(ggg.npcf_norm).reshape(window.shape)
    return kept_configurations(norm, theory, window, floor, NORM_PCT)

# Do the test for ggg
# We need 100k gals for decent convergence in normal/tight tier.
def test_ggg_natural_components_reproduce_the_closed_form(ggg_measured):
    ggg, theory, window = ggg_measured
    meas = np.asarray(ggg.npcf)[:, 0]*window
    for i in range(4):
        keep = _kept(ggg, theory[i], window, THEORY_FLOORS[i])
        thistheo = theory[i][keep]
        thismeas = meas[i][keep]
        # Assert that we retain sufficient components
        assert keep.mean() > MIN_KEPT, (i, "cuts retained only %.2f%%"%(100*keep.mean()))
        # Assert that the two amplitudes and phases agree pointwise
        assert_amp_phase(thismeas, thistheo, RTOL_NORMAL, RTOL_TIGHT, 'Gamma^%d'%i)
        # Assert that the overall complex scale is close to 1
        alpha = np.vdot(thistheo, thismeas)/np.vdot(thistheo, thistheo)
        assert abs(np.abs(alpha) - 1.) < RTOL_TIGHT, (i, alpha)
        assert abs(np.angle(alpha)) < RTOL_TIGHT, (i, alpha)
