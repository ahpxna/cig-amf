"""Fail-closed contracts for the Paper-07/09 D0 disturbance microgate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, Iterable, Mapping, Tuple


class DisturbanceRegime(str, Enum):
    RESET = "reset"
    FROZEN_POLICY = "frozen_policy"
    LIVE_LEARNING = "live_learning"


@dataclass(frozen=True)
class PairedDisturbanceRecord:
    regime: DisturbanceRegime
    arm: str
    replicate: int
    target_key: str
    immediate_cost: float
    future_state_distance: float
    future_response_shift: float
    future_policy_distance: float

    def __post_init__(self) -> None:
        if not isinstance(self.regime, DisturbanceRegime):
            raise TypeError("regime must be a DisturbanceRegime")
        if not self.arm or not self.target_key:
            raise ValueError("arm and target_key must be non-empty")
        if int(self.replicate) < 0:
            raise ValueError("replicate must be non-negative")
        values = (
            self.immediate_cost,
            self.future_state_distance,
            self.future_response_shift,
            self.future_policy_distance,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("disturbance values must be finite and non-negative")

    @property
    def future_magnitude(self) -> float:
        return max(
            float(self.future_state_distance),
            float(self.future_response_shift),
            float(self.future_policy_distance),
        )


@dataclass(frozen=True)
class DisturbanceGateResult:
    status: str
    reasons: Tuple[str, ...]
    regime_metric_means: Dict[str, Dict[str, float]]
    arm_metric_means: Dict[str, Dict[str, float]]


_METRICS = (
    "future_state_distance",
    "future_response_shift",
    "future_policy_distance",
)


def _validated_thresholds(
    values: Mapping[str, float], *, name: str
) -> Dict[str, float]:
    if set(values) != set(_METRICS):
        raise ValueError(f"{name} must define exactly {_METRICS}")
    out = {metric: float(values[metric]) for metric in _METRICS}
    if not all(math.isfinite(value) and value >= 0.0 for value in out.values()):
        raise ValueError(f"{name} values must be finite and non-negative")
    return out


def adjudicate_d0(
    records: Iterable[PairedDisturbanceRecord],
    *,
    metric_thresholds: Mapping[str, float],
    minimum_arm_spreads: Mapping[str, float],
    immediate_cost_tolerance: float,
    minimum_replicates: int = 2,
) -> DisturbanceGateResult:
    """Adjudicate existence/heterogeneity only; never claim a causal theorem.

    All reset/frozen/live cells, at least two arms, one common target key, and
    matched immediate cost are mandatory.  A pass means only that the D0 signal
    exists strongly enough to justify theorem work.
    """
    thresholds = _validated_thresholds(metric_thresholds, name="metric_thresholds")
    spreads = _validated_thresholds(
        minimum_arm_spreads, name="minimum_arm_spreads"
    )
    if not math.isfinite(float(immediate_cost_tolerance)) or float(
        immediate_cost_tolerance
    ) < 0.0:
        raise ValueError("immediate_cost_tolerance must be finite and non-negative")
    if type(minimum_replicates) is not int or minimum_replicates < 1:
        raise ValueError("minimum_replicates must be a positive integer")

    rows = tuple(records)
    if not rows:
        return DisturbanceGateResult("INCOMPLETE", ("no paired records",), {}, {})
    reasons = []
    regimes = {row.regime for row in rows}
    required = set(DisturbanceRegime)
    if regimes != required:
        reasons.append(
            "missing regimes: " + ",".join(sorted(regime.value for regime in required - regimes))
        )
    arms = sorted({row.arm for row in rows})
    if len(arms) < 2:
        reasons.append("at least two intervention arms are required")
    target_keys = {row.target_key for row in rows}
    if len(target_keys) != 1:
        reasons.append("all cells must share one fixed target_key")
    replicate_ids = sorted({int(row.replicate) for row in rows})
    if len(replicate_ids) < minimum_replicates:
        reasons.append(
            f"at least {minimum_replicates} paired replicate IDs are required"
        )
    expected_replicates = set(replicate_ids)
    for regime in DisturbanceRegime:
        for arm in arms:
            actual = {
                int(row.replicate)
                for row in rows
                if row.regime == regime and row.arm == arm
            }
            if actual != expected_replicates:
                reasons.append(
                    f"incomplete regime×arm replicates for {regime.value}/{arm}: "
                    f"expected={sorted(expected_replicates)}, actual={sorted(actual)}"
                )
    seen_cells = [(row.regime, row.arm, int(row.replicate)) for row in rows]
    if len(seen_cells) != len(set(seen_cells)):
        reasons.append("duplicate regime×arm×replicate records")

    costs = [float(row.immediate_cost) for row in rows]
    if max(costs) - min(costs) > float(immediate_cost_tolerance):
        reasons.append("immediate intervention costs are not matched")

    regime_metric_means = {
        regime.value: {
            metric: float(sum(
                float(getattr(row, metric))
                for row in rows if row.regime == regime
            ) / max(1, sum(row.regime == regime for row in rows)))
            for metric in _METRICS
        }
        for regime in DisturbanceRegime
    }
    nonreset = {
        DisturbanceRegime.FROZEN_POLICY,
        DisturbanceRegime.LIVE_LEARNING,
    }
    arm_metric_means = {
        arm: {
            metric: float(sum(
                float(getattr(row, metric))
                for row in rows if row.arm == arm and row.regime in nonreset
            ) / max(1, sum(
                row.arm == arm and row.regime in nonreset for row in rows
            )))
            for metric in _METRICS
        }
        for arm in arms
    }
    signal_metrics = []
    heterogeneous_metrics = []
    if arms:
        for metric in _METRICS:
            values = [arm_metric_means[arm][metric] for arm in arms]
            if max(values) >= thresholds[metric]:
                signal_metrics.append(metric)
            if max(values) - min(values) >= spreads[metric]:
                heterogeneous_metrics.append(metric)
    if not signal_metrics:
        reasons.append("no per-metric nonreset mean clears its calibrated threshold")
    if not heterogeneous_metrics:
        reasons.append("no per-metric nonreset arm-mean spread clears its threshold")

    status = "CONFIRMED" if not reasons else "FAILED"
    return DisturbanceGateResult(
        status=status,
        reasons=tuple(reasons),
        regime_metric_means=regime_metric_means,
        arm_metric_means=arm_metric_means,
    )
