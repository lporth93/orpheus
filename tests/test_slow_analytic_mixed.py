# Here we collect slow-tier tests that make sure the mixed correlator performs the
# correct measurement by comparing against a mock with analytically known correlators

import numpy as np

from conftest import (MIN_KEPT, RTOL_NORMAL, RTOL_TIGHT, assert_amp_phase,
                      kept_configurations)

# Per-component amplitude floors, notes sect 8.2
GNN_FLOOR, GNN_PCT = .25, 70
NGG_FLOORS, NGG_PCT = (.10, .10), 75

# TODO: Add gamma_t here?

def _kept(corr, theory, window, floor, pct):
    norm = np.asarray(corr.npcf_norm).reshape(theory.shape)
    return kept_configurations(norm, theory, window, floor, pct)

# Do the test for gnn
# We need 100k gals for decent convergence in tight tier. Needs more gals as its
# second-order in delta and therfore takes longer to converge
def test_gnn_reproduces_the_closed_form(gnn_measured):
    gnn, combined, theory, window = gnn_measured
    keep = _kept(gnn, theory, window, GNN_FLOOR, GNN_PCT)
    assert keep.mean() > MIN_KEPT, "cuts retained only %.2f%%"%(100*keep.mean())
    meas = combined*window
    assert_amp_phase(meas[keep], theory[keep], RTOL_TIGHT, RTOL_TIGHT, 'GNN')

# Do the test for gnn
# We need 60k gals for decent convergence in normal/tight tier.
def test_ngg_reproduces_the_closed_form(ngg_measured):
    ngg, combined, theory, window = ngg_measured
    meas = combined*window[None]
    for i, name in enumerate(['G_-', 'G_+']):
        keep = _kept(ngg, theory[i], window, NGG_FLOORS[i], NGG_PCT)
        assert keep.mean() > MIN_KEPT, (name, "cuts retained only %.2f%%"%(100*keep.mean()))
        assert_amp_phase(meas[i][keep], theory[i][keep], RTOL_NORMAL, RTOL_TIGHT, name)