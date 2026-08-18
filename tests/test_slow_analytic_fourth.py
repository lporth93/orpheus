# Here we collect slow-tier tests that make sure the fourth-order correlator performs the
# correct measurement by comparing against a mock with an analytically known 4pcf

import numpy as np

from conftest import (MIN_KEPT, RTOL_LOOSE, RTOL_TIGHT, assert_amp_phase,
                      kept_configurations)

# Dont consider bins which might be rather empty
NORM_PCT = 75
# Per-component amplitude floors, notes sect 8.2
THEORY_FLOORS = (.25, .15, .10, .10, .10, .05, .05, .05)

def _kept(gggg, theory, window, floor):
    return kept_configurations(np.asarray(gggg.npcf_norm)[0], theory, window, floor, NORM_PCT)

# Do the test
# We need 100k gals for decent convergence in loose tier.
def test_gggg_natural_components_reproduce_the_closed_form(gggg_measured):
    gggg, theory, window = gggg_measured
    meas = np.asarray(gggg.npcf)[:, 0]*window # Rescale measurement by window effect
    for i in range(8):
        keep = _kept(gggg, theory[i], window, THEORY_FLOORS[i])
        thistheo = theory[i][keep]
        thismeas = meas[i][keep]
        # Assert that we retain sufficient components
        assert keep.mean() > MIN_KEPT, (i, "cuts retained only %.2f%%"%(100*keep.mean()))
        # Assert that the two amplitudes and phases agree pointwise
        assert_amp_phase(thismeas, thistheo,  RTOL_LOOSE, RTOL_LOOSE, 'comp %d'%i)
        # Assert that the overall complex scale is close to 1.
        alpha = np.vdot(thistheo, thismeas)/np.vdot(thistheo, thistheo)
        assert abs(np.abs(alpha) - 1.) < RTOL_TIGHT, (i, alpha)
        assert abs(np.angle(alpha)) < RTOL_TIGHT, (i, alpha)