"""Shared latency estimand constants.

Paper/code contract
-------------------
Latency onset is a *relative* capacity-spectrum estimand, not an independently
calibrated absolute threshold:

    L_on(x) = min{ell : C^[ell](x) >= eta_L max_r C^[r](x)}

with ``eta_L = 0.05`` and a tiny numerical floor.  Oracle and learned latency
must import these constants so the estimand cannot drift between paths.
"""

LATENCY_ONSET_FRACTION = 0.05
LATENCY_ONSET_ABS_FLOOR = 1e-8
LATENCY_ONSET_RULE = (
    "first lag with capacity >= max(1e-8, 0.05 * within-sample peak capacity)"
)
