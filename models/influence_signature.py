"""
influence_signature.py — CAUSAL INFLUENCE SIGNATURES, the central novelty.

=============================================================================
ONE-SENTENCE IDEA
=============================================================================
Do not compress a neighbour's influence into ONE scalar. Retain a
MULTIDIMENSIONAL PROFILE and use it to assign ego-centric functional roles.

=============================================================================
WHY ONE SCALAR IS INSUFFICIENT
=============================================================================
Two neighbours can have the same mean |w| while being entirely different:

  Vehicle A — blocker:
      sign: strongly NEGATIVE | condition: only when ego enters the lane
      delay: IMMEDIATE
      -> w = -0.8 when ego is two cells from the lane, but w ~ 0 in another zone

  Vehicle B — relay/signaller:
      sign: POSITIVE | condition: broad | delay: 3–5 steps
      -> today's signal helps ego avoid congestion several steps later

  Vehicle C — resource consumer:
      sign: mildly negative | uniform across contexts | immediate

A and C can have the same mean |w|, yet ego must treat them DIFFERENTLY. This
is precisely the heterogeneity that a mean cannot retain: averaging discards
context-dependent sign, delay, and variability.

=============================================================================
THE FIVE SIGNATURE DIMENSIONS
=============================================================================
  0. signed_mu       — sign and magnitude of influence (help or harm)
  1. abs_mu          — pure magnitude, used for magnitude/core selection
  2. sigma           — epistemic uncertainty from ensemble disagreement
  3. temporal_std    — variation over TIME: whether j's behaviour is stable
  4. context_std     — variation across CONTEXTS: whether influence is conditional

Dimensions 3 and 4 are fundamentally different and easy to confuse:
  - high temporal_std = "strong influence today, weak tomorrow" -> UNSTABLE
  - high context_std  = "strong in the lane, absent in another zone"
                        -> CONDITIONAL INFLUENCE, a blocker property and what
                           SCIC/CAI call situation-dependent influence

=============================================================================
DISTINCTION FROM THE CLOSEST PRIOR WORK
=============================================================================
  ROMA/RODE/SIRD/LDSA/ACORM (role-based MARL):
      Roles are inferred from OBSERVATIONS such as trajectories, behaviour,
      and environmental effects, so they are correlational. Roles are GLOBAL
      and condition THAT SAME AGENT'S POLICY. Their blind spot is that two
      stationary vehicles—one blocking ego and one parked harmlessly—look
      behaviourally identical and receive the SAME role. A counterfactual
      signature separates them by asking whether ego's outcome would change
      if the neighbour acted differently.

  Jaques / SCIC / MAGIC (causal-influence MARL):
      These methods also use interventions, but convert influence into an
      INTRINSIC REWARD that answers "what should I DO?" rather than using it
      to structure a representation.

  Pieroth ICML 2024 (TIM/SIM):
      This work measures influence structure but DELIBERATELY AVOIDS
      counterfactual actions. Its max-minus-min quantity is UNSIGNED and its
      goal is DESCRIPTIVE. The Final Remarks explicitly identify using TIM/SIM
      to improve learning as FUTURE WORK.

  The open cell addressed here is:
      (SIGNED, MULTIDIMENSIONAL, EGO-CENTRIC interventional signal)
      x (MEMORY ORGANIZATION and CAPACITY ALLOCATION).
=============================================================================
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


# Number of signature dimensions.
# [SIG-5D] 6 -> 5: remove latency.
# Experimental evidence required this removal. The T4 H-sweep over
# H in {1,2,3,5,8} showed that 3/4 declared roles did NOT change sign and all
# saturated at H=2, giving a sign-flip role fraction of 0.25, below the 0.30
# threshold. All four roles had the SAME latency profile, so the arena does
# not generate a separable latency ladder. Retaining a sixth dimension would
# describe an unmeasurable quantity. The multi-horizon head is RETAINED because
# it still supplies v_tim and DR; only the signature component is removed.
# See the paper's Limitations section.
SIGNATURE_DIM = 5

# Dimension names used in the paper's centroid heat map.
SIGNATURE_NAMES = (
    "signed_mu",
    "abs_mu",
    "sigma",
    "temporal_std",
    "context_std",
)

# Role labels.
ROLE_BENEFICIAL = 0   # Strong positive influence.
ROLE_HARMFUL = 1      # Strong negative influence.
ROLE_NEUTRAL = 2      # Influence near zero.
ROLE_ANOMALOUS = 3    # High uncertainty; not yet understood.

ROLE_NAMES = ("beneficial", "harmful", "neutral", "anomalous")
ROLE_NAMES_VI = ("Thiện", "Ác", "Trung tính", "Dị biệt")

N_SEMANTIC_ROLES = 4


class InfluenceSignatureTracker:
    """
    Track a multidimensional influence profile for EVERY directed pair (i, j).

    Runner usage immediately after calling proxy.score_batch_full():

        tracker.update(
            ego_id=ego,
            neighbor_id=j,
            signed_mu=out["mu"][b],
            sigma=out["sigma"][b],
            context_key=env.agent_zone[ego],   # or any other context identifier
        )

    Subsequent access:
        sig = tracker.get_signature(ego, j)          # np [5]
        role = tracker.get_role(ego, j)              # int 0..3
        mat = tracker.get_signature_matrix(ego, ids) # np [n_ids, 5]

    Args:
        window:
            Number of recent observations retained for temporal_std.
        tau_role:
            |signed_mu| threshold separating beneficial/harmful roles from
            the neutral role.
        sigma_hi:
            Sigma threshold for the anomalous role. It should be derived from
            an observed sigma percentile, not hard-coded; use auto_calibrate().
        normalise:
            If True, get_signature_matrix returns per-dimension z-scores.
            K-means needs this because the dimensions have different scales.
    """

    def __init__(
        self,
        n_agents: int,
        window: int = 30,
        tau_role: float = 0.05,
        sigma_hi: float = 0.5,
        normalise: bool = True,
        eps: float = 1e-8,
    ):
        self.n_agents = int(n_agents)
        self.window = int(window)
        self.tau_role = float(tau_role)
        self.sigma_hi = float(sigma_hi)
        self.normalise = bool(normalise)
        self.eps = float(eps)

        # history[(i,j)] stores a deque of recent signed_mu observations.
        self._mu_hist: Dict[Tuple[int, int], deque] = {}
        self._sigma_hist: Dict[Tuple[int, int], deque] = {}

        # context_mu[(i,j)][ctx] stores mu observations from context ctx.
        self._context_mu: Dict[Tuple[int, int], Dict] = {}

        self._n_obs: Dict[Tuple[int, int], int] = {}

    # =====================================================================
    # Updates
    # =====================================================================

    def update(
        self,
        ego_id: int,
        neighbor_id: int,
        signed_mu: float,
        sigma: float,
        context_key=None,
    ):
        """
        Record one influence observation for the (ego, neighbour) pair.

        context_key:
            Any hashable context identifier: a zone ID, a coarse hash of
            relative position, or a distance bin. This supplies context_std,
            the situation-conditional dimension. If None, context_std is zero.
        """
        key = (int(ego_id), int(neighbor_id))

        if key not in self._mu_hist:
            self._mu_hist[key] = deque(maxlen=self.window)
            self._sigma_hist[key] = deque(maxlen=self.window)
            self._context_mu[key] = {}
            self._n_obs[key] = 0

        self._mu_hist[key].append(float(signed_mu))
        self._sigma_hist[key].append(float(sigma))
        self._n_obs[key] += 1

        if context_key is not None:
            ctx = self._context_mu[key]

            if context_key not in ctx:
                ctx[context_key] = deque(maxlen=self.window)

            ctx[context_key].append(float(signed_mu))

    def update_from_proxy_output(
        self,
        ego_id: int,
        neighbor_ids: List[int],
        proxy_out: Dict[str, np.ndarray],
        context_keys: Optional[List] = None,
    ):
        """
        Convenience method accepting proxy.score_batch_full() output directly.

        Args:
            neighbor_ids: list[int] of length B, ordered like the proxy batch.
            proxy_out: Dictionary from score_batch_full; requires mu and sigma.
            context_keys: List of length B, or None.
        """
        mu = np.asarray(proxy_out["mu"]).reshape(-1)            # [B]
        sigma = np.asarray(proxy_out["sigma"]).reshape(-1)      # [B]
        _unused_latency = np.asarray(
            proxy_out.get("latency", np.zeros_like(mu))
        ).reshape(-1)                                            # [B]

        for b, j in enumerate(neighbor_ids):
            self.update(
                ego_id=ego_id,
                neighbor_id=int(j),
                signed_mu=float(mu[b]),
                sigma=float(sigma[b]),
                context_key=(
                    None if context_keys is None else context_keys[b]
                ),
            )

    # =====================================================================
    # Signature retrieval
    # =====================================================================

    def get_signature(self, ego_id: int, neighbor_id: int) -> np.ndarray:
        """
        Returns:
            np.ndarray float32 shape [5] = SIGNATURE_DIM

            [0] signed_mu     signed mean
            [1] abs_mu        mean absolute magnitude
            [2] sigma         mean uncertainty
            [3] temporal_std  standard deviation of mu over time
            [4] context_std   standard deviation across context-specific mean mu
        """
        key = (int(ego_id), int(neighbor_id))

        if key not in self._mu_hist or len(self._mu_hist[key]) == 0:
            return np.zeros(SIGNATURE_DIM, dtype=np.float32)

        mus = np.asarray(self._mu_hist[key], dtype=np.float64)          # [T]
        sigmas = np.asarray(self._sigma_hist[key], dtype=np.float64)    # [T]

        signed_mu = float(np.mean(mus))

        # ---------------------------------------------------------------
        # abs_mu = mean(|mu|), NOT |mean(mu)|.
        #
        # A unit test caught this defect. Using |mean| makes this dimension
        # COMPLETELY REDUNDANT with signed_mu because it is merely its absolute
        # value, wasting one signature dimension.
        #
        # With mean(|·|), the two channels are INDEPENDENT and their ratio
        # carries real information:
        #     |signed_mu| ~ abs_mu  -> consistently one-directional influence
        #                              (a blocker always blocks; a relay helps)
        #     |signed_mu| << abs_mu -> strong but SIGN-REVERSING influence
        #                              (sometimes helpful and sometimes harmful,
        #                               precisely the anomalous-slot target)
        # This is exactly the case a scalar cannot express and why the
        # signature must be multidimensional.
        # ---------------------------------------------------------------
        abs_mu = float(np.mean(np.abs(mus)))

        sigma = float(np.mean(sigmas))
        temporal_std = float(np.std(mus)) if mus.size > 1 else 0.0

        # ---- context_std ------------------------------------------------
        # First compute a mean WITHIN EACH context, then compute the standard
        # deviation ACROSS contexts. This separates situation-dependent
        # influence from random influence noise, which temporal_std already
        # captures.
        ctx_map = self._context_mu.get(key, {})
        ctx_means = [
            float(np.mean(np.asarray(v, dtype=np.float64)))
            for v in ctx_map.values()
            if len(v) > 0
        ]

        context_std = (
            float(np.std(np.asarray(ctx_means, dtype=np.float64)))
            if len(ctx_means) > 1
            else 0.0
        )


        return np.array(
            [signed_mu, abs_mu, sigma, temporal_std, context_std],
            dtype=np.float32,
        )

    def get_signature_matrix(
        self,
        ego_id: int,
        neighbor_ids: List[int],
        normalise: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Returns:
            np.ndarray float32 shape [len(neighbor_ids), SIGNATURE_DIM]

        normalise=True applies a z-score to each COLUMN/dimension. K-means
        needs this because dimensions have different scales, for example
        abs_mu >= 0 while signed_mu is approximately in [-1, 1].
        """
        if normalise is None:
            normalise = self.normalise

        rows = [
            self.get_signature(ego_id, int(j)) for j in neighbor_ids
        ]

        if len(rows) == 0:
            return np.zeros((0, SIGNATURE_DIM), dtype=np.float32)

        mat = np.stack(rows, axis=0).astype(np.float32)  # [N, 5]

        if not normalise or mat.shape[0] < 2:
            return mat

        mean = np.mean(mat, axis=0, keepdims=True)   # [1, 5]
        std = np.std(mat, axis=0, keepdims=True)     # [1, 5]

        return ((mat - mean) / (std + self.eps)).astype(np.float32)

    def get_n_observations(self, ego_id: int, neighbor_id: int) -> int:
        return int(self._n_obs.get((int(ego_id), int(neighbor_id)), 0))

    # =====================================================================
    # ROLE ASSIGNMENT — semantic rules
    # =====================================================================

    def get_role(self, ego_id: int, neighbor_id: int) -> int:
        """
        Assign a HARD role by rule for logging, diagnostics, and slot
        initialization. Differentiable SOFT assignment is implemented by
        soft_role_assignment().

        Priority order matters: anomalous is evaluated FIRST. When a neighbour
        is not understood (high sigma), assigning it to beneficial or harmful
        is arbitrary; isolating it is safer.

        Returns:
            Integer in {0, 1, 2, 3}.
        """
        sig = self.get_signature(ego_id, neighbor_id)

        signed_mu = float(sig[0])
        sigma = float(sig[2])

        if sigma > self.sigma_hi:
            return ROLE_ANOMALOUS

        if abs(signed_mu) < self.tau_role:
            return ROLE_NEUTRAL

        return ROLE_BENEFICIAL if signed_mu > 0.0 else ROLE_HARMFUL

    def get_role_distribution(
        self, ego_id: int, neighbor_ids: List[int]
    ) -> Dict[str, int]:
        """Count neighbours by role for the paper's result tables."""
        counts = {name: 0 for name in ROLE_NAMES}

        for j in neighbor_ids:
            counts[ROLE_NAMES[self.get_role(ego_id, int(j))]] += 1

        return counts

    # =====================================================================
    # Automatic threshold calibration
    # =====================================================================

    def auto_calibrate(
        self,
        ego_ids: Optional[List[int]] = None,
        tau_percentile: float = 50.0,
        sigma_percentile: float = 80.0,
    ):
        """
        Set tau_role and sigma_hi from empirical data PERCENTILES instead of
        fixed constants.

        This is necessary because the scale of mu depends entirely on the
        environment's reward scale. Fixing tau_role = 0.05 can place EVERY
        neighbour in the neutral role when rewards are small, or NO neighbour
        there when rewards are large. Both outcomes make semantic slots useless.

        Call this once after the Stage 0 warm-up.
        """
        all_abs_mu = []
        all_sigma = []

        for key in self._mu_hist:
            if ego_ids is not None and key[0] not in ego_ids:
                continue

            sig = self.get_signature(key[0], key[1])
            all_abs_mu.append(float(sig[1]))
            all_sigma.append(float(sig[2]))

        if len(all_abs_mu) >= 4:
            self.tau_role = float(
                np.percentile(np.asarray(all_abs_mu), tau_percentile)
            )

        if len(all_sigma) >= 4:
            self.sigma_hi = float(
                np.percentile(np.asarray(all_sigma), sigma_percentile)
            )

        return {
            "tau_role": float(self.tau_role),
            "sigma_hi": float(self.sigma_hi),
            "n_pairs_used": int(len(all_abs_mu)),
        }

    # =====================================================================
    # External evaluation labels
    # =====================================================================

    def role_recovery_score(
        self,
        ego_id: int,
        neighbor_ids: List[int],
        ground_truth_roles: Dict[int, int],
    ) -> Dict[str, float]:
        """
        Compare discovered roles with the environment's TRUE roles.

        This is a key paper experiment. Even without a reward improvement,
        autonomously discovering roles that match ground truth remains a
        publishable interpretability result.

        Args:
            ground_truth_roles: {neighbor_id: role_int}

        Returns:
            Dictionary containing overall and per-role accuracy.
        """
        correct = 0
        total = 0
        per_role_correct = {r: 0 for r in range(N_SEMANTIC_ROLES)}
        per_role_total = {r: 0 for r in range(N_SEMANTIC_ROLES)}

        for j in neighbor_ids:
            j = int(j)

            if j not in ground_truth_roles:
                continue

            gt = int(ground_truth_roles[j])
            pred = self.get_role(ego_id, j)

            total += 1
            per_role_total[gt] += 1

            if pred == gt:
                correct += 1
                per_role_correct[gt] += 1

        out = {
            "accuracy": float(correct) / float(max(1, total)),
            "n_evaluated": int(total),
        }

        for r in range(N_SEMANTIC_ROLES):
            out[f"acc_{ROLE_NAMES[r]}"] = (
                float(per_role_correct[r]) / float(per_role_total[r])
                if per_role_total[r] > 0
                else float("nan")
            )

        return out

    def get_cluster_centroids(
        self,
        ego_ids: List[int],
        neighbor_ids_per_ego: Dict[int, List[int]],
    ) -> np.ndarray:
        """
        Average signatures by role, pooled across multiple ego agents.

        Returns:
            np.ndarray shape [N_SEMANTIC_ROLES, SIGNATURE_DIM]

        PAPER FIGURE: render this matrix as a heat map. Clearly different rows
        demonstrate that the slots are GENUINELY specialized rather than
        collapsed. Identical rows mean collapse remains and requires repair.
        """
        buckets = {r: [] for r in range(N_SEMANTIC_ROLES)}

        for ego in ego_ids:
            for j in neighbor_ids_per_ego.get(int(ego), []):
                r = self.get_role(int(ego), int(j))
                buckets[r].append(self.get_signature(int(ego), int(j)))

        out = np.zeros((N_SEMANTIC_ROLES, SIGNATURE_DIM), dtype=np.float32)

        for r in range(N_SEMANTIC_ROLES):
            if len(buckets[r]) > 0:
                out[r] = np.mean(
                    np.stack(buckets[r], axis=0), axis=0
                ).astype(np.float32)

        return out


# =========================================================================
# SOFT, DIFFERENTIABLE ROLE ASSIGNMENT — used by peripheral_memory_v2
# =========================================================================

def soft_role_assignment(
    signed_mu: np.ndarray,
    sigma: np.ndarray,
    tau_role: float = 0.05,
    sigma_hi: float = 0.5,
    sharpness: float = 3.0,
) -> np.ndarray:
    """
    Soft version of get_role(), returning a distribution over four roles
    instead of one hard label.

    Soft assignment is necessary because gradients cannot flow through hard
    if/else assignment. Near a boundary (mu = tau_role ± epsilon), hard labels
    also jump discontinuously and cause instability. A sigmoid with slope
    `sharpness` provides a smooth transition.

    Gate structure:
        g_anom  = sigmoid(k * (sigma - sigma_hi))        "not understood"
        g_sure  = 1 - g_anom                             "understood"
        g_pos   = sigmoid(k * (mu - tau))                "sufficiently positive"
        g_neg   = sigmoid(k * (-mu - tau))               "sufficiently negative"
        g_neu   = 1 - g_pos - g_neg  (clamp >= 0)        "near zero"

    Args:
        signed_mu: np [N]
        sigma:     np [N]

    Returns:
        float32 array of shape [N, 4], with each row summing to one.
        Columns: [beneficial, harmful, neutral, anomalous].
    """
    mu = np.asarray(signed_mu, dtype=np.float64).reshape(-1)   # [N]
    sg = np.asarray(sigma, dtype=np.float64).reshape(-1)       # [N]

    if mu.shape[0] == 0:
        return np.zeros((0, N_SEMANTIC_ROLES), dtype=np.float32)

    # -----------------------------------------------------------------
    # NORMALIZE SLOPE BY THE THRESHOLD. A unit test caught this defect.
    #
    # Directly using sigmoid(sharpness * (mu - tau)) makes the slope depend
    # on the SCALE of mu. With sharpness=10 and tau=0.05:
    #     at mu = 0, which should be completely neutral,
    #     g_pos = sigmoid(10 * (-0.05)) = sigmoid(-0.5) = 0.378  (!!)
    #     -> g_neu = 1 - 0.378 - 0.378 = 0.244 < g_pos
    #     -> mu = 0 is incorrectly assigned as beneficial. WRONG.
    #
    # Fix: divide sharpness by the threshold to make it DIMENSIONLESS, meaning
    # "the transition spans this many multiples of tau's width."
    #     k_mu = sharpness / tau  ->  at mu=0: sigmoid(-sharpness)
    # With sharpness=3: sigmoid(-3)=0.047 -> g_neu = 0.906. CORRECT.
    # -----------------------------------------------------------------
    k_mu = float(sharpness) / max(float(tau_role), 1e-8)
    k_sg = float(sharpness) / max(float(sigma_hi), 1e-8)

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    g_anom = sigmoid(k_sg * (sg - sigma_hi))       # [N]
    g_sure = 1.0 - g_anom                          # [N]

    g_pos = sigmoid(k_mu * (mu - tau_role))        # [N]
    g_neg = sigmoid(k_mu * (-mu - tau_role))       # [N]
    g_neu = np.clip(1.0 - g_pos - g_neg, 0.0, 1.0)  # [N]

    out = np.zeros((mu.shape[0], N_SEMANTIC_ROLES), dtype=np.float64)

    out[:, ROLE_BENEFICIAL] = g_sure * g_pos
    out[:, ROLE_HARMFUL] = g_sure * g_neg
    out[:, ROLE_NEUTRAL] = g_sure * g_neu
    out[:, ROLE_ANOMALOUS] = g_anom

    # Normalize to sum to one; g_pos + g_neg + g_neu may differ slightly.
    row_sum = np.sum(out, axis=1, keepdims=True)   # [N, 1]
    out = out / np.clip(row_sum, 1e-12, None)

    return out.astype(np.float32)


def kmeans_signature_clusters(
    signatures: np.ndarray,
    n_clusters: int = 4,
    n_iter: int = 25,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure-NumPy k-means over the signature matrix; sklearn is not required.

    This supports FREE-FORM SLOTS that complement fixed semantic slots and
    capture structures missed by the four hard rules.

    DESIGN NOTE: k-means alone is NOT as effective as semantic slots because
    cluster meaning CHANGES after every reclustering. The policy must relearn
    those meanings from scratch, adding exactly the source of non-stationarity
    the paper is trying to eliminate. Use a HYBRID architecture: fixed
    semantic slots plus a small number of k-means slots.

    Args:
        signatures: np [N, D], preferably already z-score normalized.

    Returns:
        labels:    np int [N]
        centroids: np float32 [n_clusters, D]
    """
    X = np.asarray(signatures, dtype=np.float64)  # [N, D]

    if X.ndim != 2 or X.shape[0] == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((n_clusters, SIGNATURE_DIM), dtype=np.float32),
        )

    N, D = X.shape
    k = int(min(n_clusters, N))

    rng = np.random.RandomState(int(seed))

    # K-means++ initialization is substantially more stable than random init.
    centroids = np.zeros((k, D), dtype=np.float64)
    centroids[0] = X[rng.randint(N)]

    for c in range(1, k):
        d2 = np.min(
            ((X[:, None, :] - centroids[None, :c, :]) ** 2).sum(axis=2),
            axis=1,
        )  # [N]

        total = float(d2.sum())

        if total <= 1e-12:
            centroids[c] = X[rng.randint(N)]
        else:
            centroids[c] = X[rng.choice(N, p=d2 / total)]

    labels = np.zeros(N, dtype=np.int64)

    for _ in range(int(n_iter)):
        # dist: [N, k]
        dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1)  # [N]

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = X[mask].mean(axis=0)

    # Pad when n_clusters > N.
    if k < n_clusters:
        pad = np.zeros((n_clusters - k, D), dtype=np.float64)
        centroids = np.concatenate([centroids, pad], axis=0)

    return labels.astype(np.int64), centroids.astype(np.float32)
