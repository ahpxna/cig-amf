"""External benchmark contract for CIG-AMF data collection.

The core algorithm only consumes a normalized population transition, a
state-dependent action mask, observable relation features, and optional
clone/restore support.  Environment-specific wrappers implement this module;
they do not leak grid, zone, or role fields into causal models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class BenchmarkCapabilities:
    """Explicit gates prevent unsupported claims from being silently run."""

    name: str
    fixed_discrete_actions: bool
    state_dependent_action_mask: bool
    clone_restore: bool
    fixed_continuation: bool
    structural_intervention: bool
    behavioural_intervention: bool
    latency_oracle: bool

    def supports(self, panel: str) -> bool:
        needs = {
            "training": ("fixed_discrete_actions", "state_dependent_action_mask", "clone_restore"),
            "h1": ("clone_restore", "fixed_continuation", "fixed_discrete_actions"),
            "h2": ("clone_restore", "structural_intervention", "behavioural_intervention"),
            "latency": ("clone_restore", "fixed_continuation", "latency_oracle"),
        }
        if panel not in needs:
            raise ValueError(f"unknown external evaluation panel: {panel}")
        return all(bool(getattr(self, key)) for key in needs[panel])


class ExternalCIGEnvironment(Protocol):
    """Normalized population environment implemented by each benchmark wrapper."""

    capabilities: BenchmarkCapabilities
    n_agents: int
    max_action_dim: int

    def reset(self, seed: int | None = None) -> Sequence[np.ndarray]: ...
    def step(self, actions: Sequence[int]) -> tuple[Sequence[np.ndarray], Sequence[float], bool, Dict[str, Any]]: ...
    def valid_action_mask(self, agent: int) -> np.ndarray: ...
    def relation_features(self, ego: int, neighbour: int) -> np.ndarray: ...
    def clone_state(self) -> Any: ...
    def restore_state(self, state: Any) -> None: ...


def flatten_observation(value: Any, width: int | None = None) -> np.ndarray:
    """Stable finite float representation for optional nested observations."""
    if value is None:
        arr = np.zeros((0,), dtype=np.float32)
    elif isinstance(value, Mapping):
        parts = [flatten_observation(value[key]) for key in sorted(value)]
        arr = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
    elif isinstance(value, (tuple, list)):
        parts = [flatten_observation(item) for item in value]
        arr = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    # External libraries may encode unavailable sensor fields as NaN/Inf.
    # Those values must never leak into policy/proxy inputs or silently poison
    # pair-representation losses.  Preserve shape while making the normalized
    # contract explicitly finite.
    arr = np.nan_to_num(
        arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0
    )
    if width is None:
        return arr
    out = np.zeros((int(width),), dtype=np.float32)
    out[: min(out.size, arr.size)] = arr[: out.size]
    return out


def require_panel(env: ExternalCIGEnvironment, panel: str) -> None:
    if not env.capabilities.supports(panel):
        raise RuntimeError(
            f"{env.capabilities.name} cannot run the {panel} panel under the "
            "CIG-AMF causal contract; do not emit that claim."
        )
    required_methods = {
        "training": ("clone_state", "restore_state"),
        "h1": ("clone_state", "restore_state", "fixed_continuation_policy", "oracle_lag_response"),
        "h2": ("clone_state", "restore_state", "apply_structural_intervention", "apply_behavioural_intervention"),
        "latency": ("clone_state", "restore_state", "fixed_continuation_policy", "oracle_lag_response"),
    }
    missing = [name for name in required_methods.get(panel, ()) if not callable(getattr(env, name, None))]
    if missing:
        raise RuntimeError(
            f"{env.capabilities.name} advertises {panel} but lacks required methods: {missing}"
        )
