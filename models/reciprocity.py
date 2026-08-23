"""reciprocity.py — Measure the reverse direction of causal influence.

=============================================================================
IDEA
=============================================================================
The rest of CIG-AMF measures one direction:

    w_ij  :  j -> i     influence of j on ego i

This module measures the reverse direction:

    g_ij  :  i -> j     influence of ego i on j

The estimator trains TWO models to predict j's next action:
    Model A — observes only j              (shadow state s_ij)
    Model B — observes j and ego i         (pair latent z_ij)

If B predicts better than A, information about ego i improves prediction of j,
indicating that j REACTS TO i. Equal performance indicates that j ignores i.

The performance difference is the information gain:

    g_ij = CE(model A) - CE(model B)     [nats]

CE is cross-entropy, or average surprise under the true action. g_ij > 0 means
conditioning on ego i reduces surprise.

=============================================================================
INFRASTRUCTURE IS ALREADY AVAILABLE — THIS IS WHY IT IS CHEAP
=============================================================================
    z_ij  is updated from (z_prev, o_i, a_i, o_j, a_j, xi)  <- with ego
    s_ij  is updated from (s_prev,        o_j, a_j, xi)     <- without ego

These representations already exist for other reasons: the shadow state
warm-starts a pair latent after promotion. Together, they form the exact
control pair required by this diagnostic. Only ONE additional linear head on
the shadow state is needed, approximately 40 lines of implementation.

=============================================================================
CONFOUNDING ISSUE AND SOLUTION
=============================================================================
Purely observational data are heavily confounded. When i and j are adjacent,
o_i and o_j observe the same context. Model B can outperform A because o_i
contains information about that shared context, not because j reacts to i.

This is the same correlation-versus-causation trap addressed by the paper.

The nearly cost-free solution is to compute the statistic only on steps where
ego i is forced to take a random action (F_i = 1). On those steps, a_i is
mechanically independent of all other variables. Any remaining predictive
value for a_j^{t+1} is therefore causal.

Epsilon forcing already covers all agents; the estimator only needs to filter
by the F_i flag.

=============================================================================
2x2 MATRIX — THIS IS THE CONCEPTUAL CONTRIBUTION
=============================================================================
Combining the two dimensions gives four types of relationships:

                     |  i->j LOW (j ignores i)     |  i->j HIGH (j reacts to i)
    -----------------|-------------------------|--------------------------
    j->i LOW        |  neutral                 |  STALKER
    (j weakly affects i)|                      |  (reacts without harming i)
    -----------------|-------------------------|--------------------------
    j->i HIGH       |  INERT OBJECT           |  STRATEGIC PAIR  (!)
    (j affects i)   |  (blocks without reacting)|

The bottom-right cell is where genuine non-stationarity arises: two agents
continually adapt to each other and form the chasing loop described as cyclic
dynamics in standard MARL. Measuring this cell directly identifies pairs
responsible for non-stationarity, a capability absent from prior work.

The bottom-left cell matters differently. Inert objects need only be avoided;
their behaviour need not be modelled because they do not react to ego.
Strategic pairs, by contrast, require careful behavioural modelling.

=============================================================================
LINK TO JAQUES (2019)
=============================================================================
Jaques social influence measures the i->j direction over ACTIONS through an
observational KL term, then uses it as intrinsic reward. Here:
    - i->j is similar but measured through a real intervention (F_i=1);
    - j->i is measured on ego return, which Jaques does not measure; and
    - the objective is relationship classification rather than reward shaping.
The Jaques quantity is therefore recovered as ONE AXIS of a two-dimensional
profile, providing a clean distinction from the closest predecessor.

=============================================================================
HONEST ASSESSMENT OF PROMISE LEVEL
=============================================================================
STRENGTHS
  + really cheap (infrastructure already exists, ~40 lines)
  + causal measurement is almost free thanks to eps-forcing already in place
  + creates a new 2x2 matrix, directly tied to the paper's central thesis
  + clean positioning compared to Jaques

RISKS
  - in cooperative resource-flow environments, most pairs may fall into the
    "inert object" cell, the strategic pair cell may be empty -> beautiful but meaningless
  - thin sample: need F_i = 1, but eps is only 3% -> very few causal samples per pair
  - unclear how the policy uses this number

CONCLUSION
  Implement this as a diagnostic first and DO NOT connect it to the decision
  loop. Plot the 2x2 matrix. Introduce the signal into core selection only if
  the matrix demonstrates real structure in a subsequent study. This retains
  the diagnostic and positioning value without making the current paper depend
  on an untested hypothesis.

  Hence, default `use_in_signature=False`.
=============================================================================
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # allow importing this file without torch
    _HAS_TORCH = False
    nn = object


# labels of the four cells
QUAD_IRRELEVANT = 0     # neutral
QUAD_FOLLOWER = 1       # stalker
QUAD_OBSTACLE = 2       # inert object
QUAD_STRATEGIC = 3      # strategic pair

QUAD_NAMES = ("irrelevant", "follower", "obstacle", "strategic")
# Compatibility alias retained for older result readers; labels are English.
QUAD_NAMES_VI = ("unrelated", "follower", "passive obstacle", "strategic coupling")


if _HAS_TORCH:

    class EgoFreeActionHead(nn.Module):
        """Model A: predicts a_j^{t+1} ONLY from shadow state (without ego).

        Keep exactly the same capacity as the head predicting from z_ij.
        Otherwise, the cross-entropy difference reflects model capacity rather
        than information about ego, invalidating the complete measurement.
        """

        def __init__(self, shadow_dim: int, action_dim: int, hidden: int = 64):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(int(shadow_dim), int(hidden)),
                nn.ReLU(),
                nn.Linear(int(hidden), int(action_dim)),
            )

        def forward(self, s: "torch.Tensor") -> "torch.Tensor":
            """s: [B, shadow_dim] -> logits [B, action_dim]"""
            return self.net(s)


class ReciprocityTracker:
    """Track information gain in both directions for each directed pair.

    Usage in runner (after having z_ij and s_ij of step t, and
    knowing the true action a_j at step t+1):

        rec.record(
            ego_id=i, neighbor_id=j,
            logits_with_ego=head_z(z_ij),        # [action_dim]
            logits_without_ego=head_shadow(s_ij),# [action_dim]
            true_next_action=a_j_next,
            ego_was_forced=bool(forced_mask_prev[i]),
        )

    Then take the result:
        g = rec.get_gain(i, j, causal_only=True)
        quad = rec.get_quadrant(i, j, w_ij=belief.debiased_mu(j))

    Args:
        window: number of recent observations to keep for each pair.
        min_causal_samples: below this threshold, causal measurement is considered unreliable and returns NaN instead of a fabricated number."""

    def __init__(
        self,
        n_agents: int,
        window: int = 200,
        min_causal_samples: int = 20,
        eps: float = 1e-8,
    ):
        self.n_agents = int(n_agents)
        self.window = int(window)
        self.min_causal_samples = int(min_causal_samples)
        self.eps = float(eps)

        # ce[(i,j)] = deque of (ce_without, ce_with, was_forced)
        self._ce: Dict[Tuple[int, int], deque] = {}

    # ------------------------------------------------------------------

    @staticmethod
    def _cross_entropy(logits: np.ndarray, target: int) -> float:
        """CE of a sample, computed in a numerically stable way (log-sum-exp).

        This is predictive surprise: CE increases as the probability assigned
        to the true action decreases.
        """
        z = np.asarray(logits, dtype=np.float64).reshape(-1)
        z = z - np.max(z)
        logZ = float(np.log(np.sum(np.exp(z)) + 1e-300))
        t = int(np.clip(target, 0, z.shape[0] - 1))

        return float(logZ - z[t])

    def record(
        self,
        ego_id: int,
        neighbor_id: int,
        logits_with_ego,
        logits_without_ego,
        true_next_action: int,
        ego_was_forced: bool = False,
    ):
        """Record an observation for the pair (i, j)."""
        key = (int(ego_id), int(neighbor_id))

        if key not in self._ce:
            self._ce[key] = deque(maxlen=self.window)

        if _HAS_TORCH and isinstance(logits_with_ego, torch.Tensor):
            logits_with_ego = logits_with_ego.detach().cpu().numpy()

        if _HAS_TORCH and isinstance(logits_without_ego, torch.Tensor):
            logits_without_ego = logits_without_ego.detach().cpu().numpy()

        ce_with = self._cross_entropy(logits_with_ego, true_next_action)
        ce_without = self._cross_entropy(logits_without_ego, true_next_action)

        self._ce[key].append(
            (float(ce_without), float(ce_with), bool(ego_was_forced))
        )

    # ------------------------------------------------------------------

    def get_gain(
        self,
        ego_id: int,
        neighbor_id: int,
        causal_only: bool = True,
    ) -> float:
        """Information gain g_ij = CE_without - CE_with, unit nats.

        Args:
            causal_only:
                True uses ONLY steps where ego's action was forced. On those
                steps, a_i is independent of other variables, making the
                predictive signal causal. Insufficient samples return NaN
                instead of a fabricated estimate.
                False uses every step. This is inexpensive and nearly always
                produces a number, but is CONFOUNDED because o_i and o_j
                observe the same world.

        Returns:
            A float, possibly NaN when causal_only has insufficient samples.
        """
        key = (int(ego_id), int(neighbor_id))
        rows = self._ce.get(key)

        if not rows:
            return float("nan") if causal_only else 0.0

        if causal_only:
            sel = [(a, b) for a, b, f in rows if f]

            if len(sel) < self.min_causal_samples:
                return float("nan")
        else:
            sel = [(a, b) for a, b, _ in rows]

        arr = np.asarray(sel, dtype=np.float64)      # [n, 2]

        return float(np.mean(arr[:, 0] - arr[:, 1]))

    def get_n_causal_samples(self, ego_id: int, neighbor_id: int) -> int:
        rows = self._ce.get((int(ego_id), int(neighbor_id)))

        if not rows:
            return 0

        return int(sum(1 for _, _, f in rows if f))

    # ------------------------------------------------------------------

    def get_quadrant(
        self,
        ego_id: int,
        neighbor_id: int,
        w_ij: float,
        tau_w: float = 0.05,
        tau_g: float = 0.02,
        causal_only: bool = True,
    ) -> int:
        """Place the pair (i, j) into one of the four cells.

        Args:
            w_ij: Influence j->i from belief.debiased_mu(j).
            tau_w, tau_g: Thresholds for the two axes. Derive these from the
                observed-data distribution rather than hard-coding them; see
                calibrate().

        Returns:
            Integer in {0, 1, 2, 3}, or -1 for insufficient causal data.
        """
        g = self.get_gain(ego_id, neighbor_id, causal_only=causal_only)

        if not np.isfinite(g):
            return -1

        strong_in = abs(float(w_ij)) > float(tau_w)   # j influences i.
        strong_out = float(g) > float(tau_g)          # i influences j.

        if strong_in and strong_out:
            return QUAD_STRATEGIC

        if strong_in:
            return QUAD_OBSTACLE

        if strong_out:
            return QUAD_FOLLOWER

        return QUAD_IRRELEVANT

    def calibrate(
        self,
        percentile: float = 70.0,
        causal_only: bool = True,
    ) -> Dict[str, float]:
        """Set the threshold tau_g based on the distribution of the observed data.

        This is necessary because information-gain scale depends on action-space
        entropy. With five actions, maximum gain is log(5) ~ 1.61 nats.
        A hard-coded 0.02 may be appropriate here and entirely wrong elsewhere.
        """
        vals = []

        for (i, j) in self._ce:
            g = self.get_gain(i, j, causal_only=causal_only)

            if np.isfinite(g):
                vals.append(g)

        if len(vals) < 4:
            return {"tau_g": 0.02, "n_pairs": len(vals), "calibrated": False}

        return {
            "tau_g": float(np.percentile(np.asarray(vals), percentile)),
            "n_pairs": int(len(vals)),
            "mean_gain": float(np.mean(vals)),
            "max_gain": float(np.max(vals)),
            "calibrated": True,
        }

    # ------------------------------------------------------------------

    def quadrant_report(
        self,
        belief_modules: Dict,
        tau_w: float = 0.05,
        tau_g: float = 0.02,
        causal_only: bool = True,
    ) -> Dict:
        """Scan all pairs, count each cell, and return data for plotting.

        PAPER FIGURE: scatter plot with x-axis w_ij (j->i) and y-axis g_ij
        (i->j), divided into four quadrants by the two thresholds. Clear
        clustering into multiple cells demonstrates real structure worth
        incorporating into the mechanism. Concentration in one cell means the
        diagnostic adds no information and should be reported directly in the
        Discussion.

        Returns:
            Dictionary containing counts, plot points, and reliable-sample
            coverage.
        """
        counts = {name: 0 for name in QUAD_NAMES}
        counts["insufficient_data"] = 0

        points = []

        for ego_id, mod in belief_modules.items():
            for j in getattr(mod, "neighbor_ids", []):
                if int(ego_id) == int(j):
                    continue

                w = float(mod.debiased_mu(int(j)))
                g = self.get_gain(int(ego_id), int(j), causal_only=causal_only)

                q = self.get_quadrant(
                    int(ego_id), int(j), w_ij=w,
                    tau_w=tau_w, tau_g=tau_g, causal_only=causal_only,
                )

                if q < 0:
                    counts["insufficient_data"] += 1
                    continue

                counts[QUAD_NAMES[q]] += 1
                points.append({
                    "ego": int(ego_id), "neighbor": int(j),
                    "w_ij": w, "g_ij": float(g), "quadrant": int(q),
                })

        total = max(1, sum(counts.values()))

        return {
            "counts": counts,
            "points": points,
            "n_pairs": int(total),
            "coverage": float(1.0 - counts["insufficient_data"] / total),
            "tau_w": float(tau_w),
            "tau_g": float(tau_g),
        }

    def get_diagnostics(self) -> Dict:
        n_pairs = len(self._ce)
        n_causal = [
            self.get_n_causal_samples(i, j) for (i, j) in self._ce
        ]

        return {
            "n_pairs_tracked": int(n_pairs),
            "mean_causal_samples": (
                float(np.mean(n_causal)) if n_causal else 0.0
            ),
            "min_causal_samples_seen": int(min(n_causal)) if n_causal else 0,
            "pairs_with_enough_causal": int(
                sum(1 for c in n_causal if c >= self.min_causal_samples)
            ),
        }
