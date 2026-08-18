"""Third-order natural components against the closed form.

Closed forms: notes eq (gamma0)/(gammak), centroid projection of notes eq (proj). How the
cuts and the tolerance tiers are chosen, and why: notes sect 8.2.

One convention detail is specific to this file. ``multipoles2npcf`` evaluates the truncated
Fourier sum at each ``phi`` and ``_x2centroid_ggg`` rotates at that same ``phi``, so the
angular grid contributes nothing; the radial bins are averaged over, however, while a single
centroid rotation is applied at ``bin_centers_mean``. With no sampling at all that costs
6.0e-4 rms at ``binsize = 0.1``, falling by four for every halving -- a bin average taken at
the bin centre cancels the linear term, so the error is quadratic.
"""

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
    """Measured at ngal = 100k, binsize = 0.1: amplitude 0.5-0.9%, phase 0.5-0.6e-2 rad,
    overall complex scale |alpha| = 0.9982-0.9984 with arg ~1e-6.

    The window is restored on the measurement side rather than divided into the theory:
    ``1/f3`` would amplify exactly the configurations with the fewest triplets.

    Both a pointwise and an overall-scale check are made. A pointwise tolerance wide enough
    for the tail of the kept configurations would hide a percent-level error in the overall
    normalisation, which is what a wrong window or a wrong projection produces.
    ``alpha = <theory,measured>/<theory,theory>`` is used rather than a ratio of peaks
    because noise leaves it unbiased. It is weighted towards the largest bins, which is the
    price of that unbiasedness.

    Note that ``Gamma^(0)`` is *not* the largest component: it carries
    ``(beta_0 beta_1 beta_2)^2`` and so vanishes whenever a vertex approaches the centroid,
    while each ``Gamma^(k)`` keeps a term in ``beta_k^2 + 2 beta_l beta_m`` that does not.
    """
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
