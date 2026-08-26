"""
influence_signature.py — typed CIG-AMF influence-profile transport.

=============================================================================
ONE-SENTENCE IDEA
=============================================================================
Do not compress a learned interventional response surface into one scalar too
early. Transport structural capacity, behavioural direction, separate
uncertainties, contextuality, and validity-masked response latency to the
architecture, while keeping the allocator projection distinct from the full
retained profile.

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
  0. C               — standardised structural causal capacity
  1. D               — current-policy behavioural direction
  2. sigma_C         — uncertainty of capacity
  3. sigma_D         — uncertainty of direction
  4. v_ctx           — variation of capacity across contexts

Dimensions 3 and 4 are fundamentally different and easy to confuse:
  - high temporal_std = "strong influence today, weak tomorrow" -> UNSTABLE
  - high context_std  = "strong in the lane, absent in another zone"
                        -> CONDITIONAL INFLUENCE, a blocker property and what
                           SCIC/CAI call situation-dependent influence

=============================================================================
SCIENTIFIC POSITIONING
=============================================================================
Influence measurement, multi-step causal influence, dynamic interaction graphs,
and role discovery are prior art.  The paper-level claim is narrower: CIG-AMF
derives task-specific coordinates from a shared interventional response surface
and assigns them different architectural jobs.  C controls expensive-capacity
allocation; D controls policy-contrast semantics; latency remains typed temporal
information.  This module implements that transport contract and does not claim
that causal influence estimation itself is novel.
=============================================================================
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# Paper B's allocator vector is the five-dimensional structural/behavioural
# projection of its retained eight-coordinate profile in Eq. (2)--(3).
# Allocation consumes only this projection; representation consumes the
# complete typed profile below, including context and latency masks.
SIGNATURE_DIM = 5
SIGNATURE_NAMES = ("capacity", "direction", "sigma_capacity", "sigma_direction", "context_std")


@dataclass(frozen=True)
class CausalPairSignal:
    """Typed Paper-A signal transported to Paper-B without magic sentinels."""

    ego_id: int
    target_id: int
    timestamp: int
    structure_regime_id: int
    capacity: float
    direction: float
    sigma_capacity: float
    sigma_direction: float
    context_variation: float
    context_valid: bool
    latency_onset: int
    latency_peak: int
    latency_cm: float
    latency_valid: bool
    latency_onset_valid: bool
    support_valid: bool
    valid_action_count: int
    latency_horizon: int
    estimator_version: int = 1

    def __post_init__(self):
        """Enforce the typed boundary's validity-mask/sentinel contract."""
        nonnegative = {
            "capacity": self.capacity,
            "sigma_capacity": self.sigma_capacity,
            "sigma_direction": self.sigma_direction,
            "context_variation": self.context_variation,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(float(self.direction)):
            raise ValueError("direction must be finite")
        if int(self.valid_action_count) < 0:
            raise ValueError("valid_action_count must be non-negative")
        if int(self.latency_horizon) < 0:
            raise ValueError("latency_horizon must be non-negative")
        # With one valid action, C=max(Q)-min(Q)=0 but there is no treatment
        # contrast. Such a row cannot be support-valid for capacity ranking,
        # latency, or C/D pair supervision.
        if bool(self.support_valid) and int(self.valid_action_count) < 2:
            raise ValueError(
                "support_valid requires at least two state-valid target actions"
            )
        if bool(self.latency_valid):
            if int(self.latency_horizon) < 2:
                raise ValueError(
                    "valid normalized latency requires a horizon of at least two"
                )
            if (
                int(self.latency_peak) < 0
                or int(self.latency_peak) >= int(self.latency_horizon)
                or not np.isfinite(float(self.latency_cm))
                or float(self.latency_cm) < 0.0
                or float(self.latency_cm) > float(self.latency_horizon - 1)
            ):
                raise ValueError(
                    "valid latency requires an in-horizon peak and finite CM"
                )
        else:
            if int(self.latency_peak) != -1 or float(self.latency_cm) != 0.0:
                raise ValueError(
                    "invalid latency must use peak=-1 and CM=0"
                )
        if bool(self.latency_onset_valid):
            if (
                not bool(self.latency_valid)
                or int(self.latency_onset) < 0
                or int(self.latency_onset) >= int(self.latency_horizon)
            ):
                raise ValueError(
                    "valid latency onset requires valid latency and onset>=0"
                )
        elif int(self.latency_onset) != -1:
            raise ValueError("invalid latency onset must use onset=-1")

    # Paper notation aliases keep the typed boundary self-documenting while
    # retaining Pythonic field names for internal callers.
    @property
    def C(self) -> float:
        return float(self.capacity)

    @property
    def D(self) -> float:
        return float(self.direction)

    @property
    def sigma_C(self) -> float:
        return float(self.sigma_capacity)

    @property
    def sigma_D(self) -> float:
        return float(self.sigma_direction)

    @property
    def v_ctx(self) -> float:
        return float(self.context_variation)

    @property
    def allocator_profile(self) -> np.ndarray:
        return np.asarray(
            [
                self.capacity,
                self.direction,
                self.sigma_capacity,
                self.sigma_direction,
                self.context_variation,
            ],
            dtype=np.float32,
        )

    @property
    def normalized_latency(self) -> float:
        """Return Paper-B's ``L_cm/(H-1)`` only when latency is supported."""
        if not bool(self.latency_representation_valid):
            return 0.0
        return float(self.latency_cm) / float(int(self.latency_horizon) - 1)

    @property
    def latency_representation_valid(self) -> bool:
        """Whether latency has both a spectrum and current action support."""
        return bool(self.latency_valid and self.support_valid)

    @property
    def retained_profile(self) -> np.ndarray:
        """Return Paper B Eq. (3): ``[C,D,sC,sD,vctx,mctx,L~,mL]``."""
        return np.asarray(
            [
                self.capacity,
                self.direction,
                self.sigma_capacity,
                self.sigma_direction,
                self.context_variation,
                float(bool(self.context_valid)),
                self.normalized_latency,
                float(bool(self.latency_representation_valid)),
            ],
            dtype=np.float32,
        )

# Role labels.
ROLE_BENEFICIAL = 0   # Strong positive influence.
ROLE_HARMFUL = 1      # Strong negative influence.
ROLE_NEUTRAL = 2      # Influence near zero.
ROLE_UNCERTAIN = 3    # High ensemble disagreement, not an anomaly claim.

ROLE_NAMES = ("beneficial", "harmful", "neutral", "uncertain")
ROLE_NAMES_VI = ROLE_NAMES

N_SEMANTIC_ROLES = 4


class InfluenceSignatureTracker:
    """
    Track a multidimensional influence profile for EVERY directed pair (i, j).

    Runner usage records distinct structural and behavioural coordinates:

        tracker.update(
            ego_id=ego, neighbor_id=j,
            capacity=out["c_mu"][b], direction=out["d_mu"][b],
            sigma_capacity=out["c_sigma"][b],
            sigma_direction=out["d_sigma"][b],
            context_key=adapter.context_key(ego, j),
        )

    Subsequent access:
        sig = tracker.get_signature(ego, j)          # np [5]
        role = tracker.get_role(ego, j)              # int 0..3
        mat = tracker.get_signature_matrix(ego, ids) # np [n_ids, 5]

    Args:
        window:
            Number of recent observations retained for temporal_std.
        tau_role:
        |direction| threshold separating beneficial/harmful roles from
            the neutral role.
        sigma_hi:
            Sigma threshold for the uncertain route. It should be derived from
            an observed sigma percentile, not hard-coded; use auto_calibrate().
        normalise:
            If True, get_signature_matrix returns per-dimension z-scores.
            K-means needs this because the dimensions have different scales.
    """

    def __init__(
        self,
        n_agents: int,
        window: int = 30,
        direction_window: int = 5,
        tau_role: float = 0.05,
        sigma_hi: float = 0.5,
        normalise: bool = True,
        eps: float = 1e-8,
        allow_legacy_direction_fallback: bool = False,
    ):
        self.n_agents = int(n_agents)
        self.window = int(window)
        self.direction_window = int(max(1, direction_window))
        self.tau_role = float(tau_role)
        self.sigma_hi = float(sigma_hi)
        self.normalise = bool(normalise)
        self.eps = float(eps)
        self.allow_legacy_direction_fallback = bool(allow_legacy_direction_fallback)

        # C and D must remain distinct: C tracks standardised structural
        # capacity, while D tracks the current execution policy's direction.
        self._capacity_hist: Dict[Tuple[int, int], deque] = {}
        self._direction_hist: Dict[Tuple[int, int], deque] = {}
        self._sigma_capacity_hist: Dict[Tuple[int, int], deque] = {}
        self._sigma_direction_hist: Dict[Tuple[int, int], deque] = {}

        # Contextuality is defined on C, not policy-dependent D.
        self._context_capacity: Dict[Tuple[int, int], Dict] = {}
        self._latency_hist: Dict[Tuple[int, int], deque] = {}
        self._latency_valid_hist: Dict[Tuple[int, int], deque] = {}
        self._latency_onset_hist: Dict[Tuple[int, int], deque] = {}
        self._latency_onset_valid_hist: Dict[Tuple[int, int], deque] = {}
        self._latency_peak_hist: Dict[Tuple[int, int], deque] = {}
        self._latency_horizon_hist: Dict[Tuple[int, int], deque] = {}

        self._n_obs: Dict[Tuple[int, int], int] = {}

    def reconcile_candidate_pairs(self, candidate_neighbors_by_ego):
        """Evict live signature buffers for pairs outside the current candidates."""
        active = {
            (int(ego), int(j))
            for ego, ids in dict(candidate_neighbors_by_ego or {}).items()
            for j in ids
            if int(ego) != int(j)
        }
        stores = (
            self._capacity_hist, self._direction_hist,
            self._sigma_capacity_hist, self._sigma_direction_hist,
            self._context_capacity, self._latency_hist,
            self._latency_valid_hist, self._latency_onset_hist,
            self._latency_onset_valid_hist, self._latency_peak_hist,
            self._latency_horizon_hist, self._n_obs,
        )
        removed = set()
        for store in stores:
            for key in list(store.keys()):
                if key not in active:
                    removed.add(key)
                    store.pop(key, None)
        return removed

    # =====================================================================
    # Updates
    # =====================================================================

    def update(
        self,
        ego_id: int,
        neighbor_id: int,
        signed_mu: Optional[float] = None,
        sigma: Optional[float] = None,
        context_key=None,
        *,
        capacity: Optional[float] = None,
        direction: Optional[float] = None,
        sigma_capacity: Optional[float] = None,
        sigma_direction: Optional[float] = None,
        latency_center: Optional[float] = None,
        latency_valid: Optional[float] = None,
        latency_onset: Optional[int] = None,
        latency_peak: Optional[int] = None,
        latency_onset_valid: Optional[float] = None,
        latency_horizon: Optional[int] = None,
    ):
        """
        Record one C/D response-profile observation for the pair.

        ``signed_mu``/``sigma`` are legacy aliases for direction and its
        uncertainty. They are accepted only for explicit compatibility paths;
        Paper A/B launchers must provide separate C and D coordinates.

        context_key:
            Any hashable context identifier: a zone ID, a coarse hash of
            relative position, or a distance bin. This supplies context_std,
            the situation-conditional dimension. If None, context_std is zero.
        """
        key = (int(ego_id), int(neighbor_id))

        if capacity is None or direction is None:
            if not self.allow_legacy_direction_fallback:
                raise ValueError(
                    "Final CIG-AMF requires distinct capacity C and direction D; "
                    "legacy C=|D| conversion is disabled"
                )
            capacity = abs(float(0.0 if signed_mu is None else signed_mu))
            direction = float(0.0 if signed_mu is None else signed_mu)
        if sigma_capacity is None:
            sigma_capacity = float(0.0 if sigma is None else sigma)
        if sigma_direction is None:
            sigma_direction = float(0.0 if sigma is None else sigma)

        for name, value in (
            ("capacity", capacity),
            ("sigma_capacity", sigma_capacity),
            ("sigma_direction", sigma_direction),
            ("direction", direction),
        ):
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(capacity) < 0.0 or float(sigma_capacity) < 0.0 or float(sigma_direction) < 0.0:
            raise ValueError("capacity and uncertainties must be non-negative")

        if key not in self._capacity_hist:
            self._capacity_hist[key] = deque(maxlen=self.window)
            self._direction_hist[key] = deque(maxlen=self.direction_window)
            self._sigma_capacity_hist[key] = deque(maxlen=self.window)
            self._sigma_direction_hist[key] = deque(maxlen=self.direction_window)
            self._context_capacity[key] = {}
            self._latency_hist[key] = deque(maxlen=self.window)
            self._latency_valid_hist[key] = deque(maxlen=self.window)
            self._latency_onset_hist[key] = deque(maxlen=self.window)
            self._latency_peak_hist[key] = deque(maxlen=self.window)
            self._latency_onset_valid_hist[key] = deque(maxlen=self.window)
            self._latency_horizon_hist[key] = deque(maxlen=self.window)
            self._n_obs[key] = 0

        self._capacity_hist[key].append(max(0.0, float(capacity)))
        self._direction_hist[key].append(float(direction))
        self._sigma_capacity_hist[key].append(max(0.0, float(sigma_capacity)))
        self._sigma_direction_hist[key].append(max(0.0, float(sigma_direction)))
        horizon = int(0 if latency_horizon is None else latency_horizon)
        if horizon < 0:
            raise ValueError("latency_horizon must be non-negative")
        # H=1 can estimate a one-step response but cannot define L_cm/(H-1).
        valid_latency = bool(latency_valid) and horizon >= 2
        latency_value = float(-1.0 if latency_center is None else latency_center)
        if valid_latency and not np.isfinite(latency_value):
            raise ValueError("valid latency_center must be finite")
        self._latency_valid_hist[key].append(float(valid_latency))
        self._latency_hist[key].append(latency_value if valid_latency else -1.0)
        self._latency_onset_hist[key].append(
            int(-1 if latency_onset is None or not valid_latency else latency_onset)
        )
        self._latency_peak_hist[key].append(
            int(-1 if latency_peak is None or not valid_latency else latency_peak)
        )
        self._latency_horizon_hist[key].append(horizon)
        onset_valid = (
            bool(latency_onset_valid)
            if latency_onset_valid is not None
            else (latency_onset is not None and int(latency_onset) >= 0)
        )
        if onset_valid and (not valid_latency or latency_onset is None or int(latency_onset) < 0):
            raise ValueError("valid latency onset requires valid latency and onset>=0")
        if valid_latency and (latency_peak is None or int(latency_peak) < 0):
            raise ValueError("valid latency requires peak>=0")
        self._latency_onset_valid_hist[key].append(float(onset_valid))
        self._n_obs[key] += 1

        if context_key is not None:
            ctx = self._context_capacity[key]

            if context_key not in ctx:
                ctx[context_key] = deque(maxlen=self.window)

            ctx[context_key].append(max(0.0, float(capacity)))

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
        if "d_mu" in proxy_out:
            d_mu = np.asarray(proxy_out["d_mu"])
        elif self.allow_legacy_direction_fallback and "mu" in proxy_out:
            d_mu = np.asarray(proxy_out["mu"])
        else:
            raise ValueError("proxy output must provide distinct d_mu and c_mu")
        if "d_sigma" in proxy_out:
            d_sigma = np.asarray(proxy_out["d_sigma"])
        elif self.allow_legacy_direction_fallback and "sigma" in proxy_out:
            d_sigma = np.asarray(proxy_out["sigma"])
        else:
            raise ValueError("proxy output must provide distinct d_sigma and c_sigma")
        if "c_mu" in proxy_out:
            c_mu = np.asarray(proxy_out["c_mu"])
        elif self.allow_legacy_direction_fallback and "mu_range" in proxy_out:
            c_mu = np.asarray(proxy_out["mu_range"])
        else:
            raise ValueError("proxy output must provide structural capacity c_mu")
        if "c_sigma" in proxy_out:
            c_sigma = np.asarray(proxy_out["c_sigma"])
        elif self.allow_legacy_direction_fallback:
            c_sigma = d_sigma
        else:
            raise ValueError("proxy output must provide structural uncertainty c_sigma")
        d_mu = d_mu.reshape(-1)
        d_sigma = d_sigma.reshape(-1)
        c_mu = c_mu.reshape(-1)
        c_sigma = c_sigma.reshape(-1)
        latency_center = np.asarray(
            proxy_out.get("latency_center", np.full_like(c_mu, -1.0))
        ).reshape(-1)
        latency_valid = np.asarray(
            proxy_out.get("latency_valid", np.zeros_like(c_mu))
        ).reshape(-1)
        latency_onset_valid = np.asarray(
            proxy_out.get("latency_onset_valid", np.zeros_like(c_mu))
        ).reshape(-1)
        horizon = int(np.asarray(
            proxy_out.get("c_lag_mu", np.zeros((1, 1)))
        ).shape[-1])

        for b, j in enumerate(neighbor_ids):
            self.update(
                ego_id=ego_id,
                neighbor_id=int(j),
                capacity=float(c_mu[b]),
                direction=float(d_mu[b]),
                sigma_capacity=float(c_sigma[b]),
                sigma_direction=float(d_sigma[b]),
                latency_center=(
                    float(latency_center[b])
                ),
                latency_valid=float(latency_valid[b]),
                latency_onset=int(np.asarray(
                    proxy_out.get("latency_onset", np.full_like(c_mu, -1))
                ).reshape(-1)[b]),
                latency_peak=int(np.asarray(
                    proxy_out.get("latency_peak", np.full_like(c_mu, -1))
                ).reshape(-1)[b]),
                latency_onset_valid=float(latency_onset_valid[b]),
                latency_horizon=horizon,
                context_key=(
                    None if context_keys is None else context_keys[b]
                ),
            )

    # =====================================================================
    # Signature retrieval
    # =====================================================================

    def get_signature(self, ego_id: int, neighbor_id: int) -> np.ndarray:
        """
        Returns the Paper B allocator view ``[C, D, sigma_C, sigma_D, v_ctx]``.
        The typed :class:`CausalPairSignal` retains context/latency masks.
        """
        key = (int(ego_id), int(neighbor_id))

        if key not in self._capacity_hist or len(self._capacity_hist[key]) == 0:
            return np.zeros(SIGNATURE_DIM, dtype=np.float32)

        capacities = np.asarray(self._capacity_hist[key], dtype=np.float64)
        directions = np.asarray(self._direction_hist[key], dtype=np.float64)
        sigmas_c = np.asarray(self._sigma_capacity_hist[key], dtype=np.float64)
        sigmas_d = np.asarray(self._sigma_direction_hist[key], dtype=np.float64)

        capacity = float(np.mean(capacities))
        direction = float(np.mean(directions))
        sigma_capacity = float(np.mean(sigmas_c))
        sigma_direction = float(np.mean(sigmas_d))

        # ---- context_std ------------------------------------------------
        # First compute a mean WITHIN EACH context, then compute the standard
        # deviation ACROSS contexts. This separates situation-dependent
        # influence from random influence noise, which temporal_std already
        # captures.
        ctx_map = self._context_capacity.get(key, {})
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
            [
                capacity,
                direction,
                sigma_capacity,
                sigma_direction,
                context_std,
            ],
            dtype=np.float32,
        )

    def get_pair_signal(
        self,
        ego_id: int,
        neighbor_id: int,
        *,
        timestamp: int = -1,
        structure_regime_id: int = 0,
        support_valid: bool = True,
        valid_action_count: int = 0,
    ) -> CausalPairSignal:
        """Return the typed C/D/latency signal consumed across the boundary."""
        key = (int(ego_id), int(neighbor_id))
        profile = self.get_signature(*key)
        # Latency is an action-time measurement, not a slow averaged belief.
        # Export the latest response spectrum and its masks together.  Using
        # ``any``/a window average here would incorrectly make a now-invalid
        # signal appear valid because an older update happened to be valid.
        valid_hist = self._latency_valid_hist.get(key, ())
        onset_valid_hist = self._latency_onset_valid_hist.get(key, ())
        center_hist = self._latency_hist.get(key, ())
        onset_hist = self._latency_onset_hist.get(key, ())
        peak_hist = self._latency_peak_hist.get(key, ())
        horizon_hist = self._latency_horizon_hist.get(key, ())
        valid = bool(valid_hist and bool(valid_hist[-1]))
        onset_valid = bool(onset_valid_hist and bool(onset_valid_hist[-1]))
        onset = int(onset_hist[-1]) if onset_hist else -1
        peak = int(peak_hist[-1]) if peak_hist else -1
        center = float(center_hist[-1]) if center_hist else 0.0
        horizon = int(horizon_hist[-1]) if horizon_hist else 0
        return CausalPairSignal(
            ego_id=int(ego_id), target_id=int(neighbor_id),
            timestamp=int(timestamp), structure_regime_id=int(structure_regime_id),
            capacity=float(profile[0]), direction=float(profile[1]),
            sigma_capacity=float(profile[2]), sigma_direction=float(profile[3]),
            context_variation=float(profile[4]), context_valid=bool(self.get_context_validity(*key)),
            latency_onset=onset if onset_valid else -1,
            latency_peak=peak if valid else -1,
            latency_cm=center if valid else 0.0,
            latency_valid=valid, latency_onset_valid=onset_valid,
            support_valid=bool(support_valid),
            valid_action_count=int(valid_action_count),
            latency_horizon=horizon,
        )

    def get_context_validity(
        self,
        ego_id: int,
        neighbor_id: int,
        min_contexts: int = 2,
        min_samples_per_context: int = 2,
    ) -> float:
        """Return one only when contextuality has sufficient support."""
        key = (int(ego_id), int(neighbor_id))
        supported = [
            values
            for values in self._context_capacity.get(key, {}).values()
            if len(values) >= int(min_samples_per_context)
        ]
        return float(len(supported) >= int(min_contexts))

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
        capacity is non-negative while direction is signed.
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

        Priority order matters: uncertainty is evaluated FIRST. When a neighbour
        is not understood (high sigma), assigning it to beneficial or harmful
        is arbitrary; isolating it is safer.

        Returns:
            Integer in {0, 1, 2, 3}.
        """
        sig = self.get_signature(ego_id, neighbor_id)

        direction = float(sig[1])
        sigma_direction = float(sig[3])

        if sigma_direction > self.sigma_hi:
            return ROLE_UNCERTAIN

        if abs(direction) < self.tau_role:
            return ROLE_NEUTRAL

        return ROLE_BENEFICIAL if direction > 0.0 else ROLE_HARMFUL

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

        for key in self._capacity_hist:
            if ego_ids is not None and key[0] not in ego_ids:
                continue

            sig = self.get_signature(key[0], key[1])
            all_abs_mu.append(abs(float(sig[1])))
            all_sigma.append(float(sig[3]))

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
    temperature_D: float = 0.05,
    temperature_0: float = 0.05,
    temperature_sigma: float = 0.05,
) -> np.ndarray:
    """
    Soft version of get_role(), returning a distribution over four roles
    instead of one hard label.

    This diagnostic mirror uses the same Paper-B router as the runtime
    module.  Direction determines beneficial/harmful/neutral valence and
    epistemic disagreement determines the separate uncertain route:

    ``a=sigmoid((sigma_D-sigma_hi)/T_sigma)`` and
    ``v=softmax([(D-tau_D)/T_D, (-D-tau_D)/T_D,
                 (tau_D-|D|)/T_0])``.

    Returned mass is ``[(1-a)v+, (1-a)v-, (1-a)v0, a]``.

    Args:
        signed_mu: np [N]
        sigma:     np [N]

    Returns:
        float32 array of shape [N, 4], with each row summing to one.
        Columns: [beneficial, harmful, neutral, uncertain].
    """
    mu = np.asarray(signed_mu, dtype=np.float64).reshape(-1)   # [N]
    sg = np.asarray(sigma, dtype=np.float64).reshape(-1)       # [N]

    if mu.shape[0] == 0:
        return np.zeros((0, N_SEMANTIC_ROLES), dtype=np.float32)

    if min(float(temperature_D), float(temperature_0), float(temperature_sigma)) <= 0.0:
        raise ValueError("semantic routing temperatures must be positive")
    logits = np.stack([
        (mu - float(tau_role)) / float(temperature_D),
        (-mu - float(tau_role)) / float(temperature_D),
        (float(tau_role) - np.abs(mu)) / float(temperature_0),
    ], axis=1)
    logits -= np.max(logits, axis=1, keepdims=True)
    valence = np.exp(np.clip(logits, -60.0, 60.0))
    valence /= np.clip(np.sum(valence, axis=1, keepdims=True), 1e-12, None)
    uncertain = 1.0 / (1.0 + np.exp(-np.clip(
        (sg - float(sigma_hi)) / float(temperature_sigma), -60.0, 60.0
    )))
    certain = 1.0 - uncertain
    out = np.zeros((mu.shape[0], N_SEMANTIC_ROLES), dtype=np.float64)
    out[:, ROLE_BENEFICIAL] = certain * valence[:, 0]
    out[:, ROLE_HARMFUL] = certain * valence[:, 1]
    out[:, ROLE_NEUTRAL] = certain * valence[:, 2]
    out[:, ROLE_UNCERTAIN] = uncertain

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
