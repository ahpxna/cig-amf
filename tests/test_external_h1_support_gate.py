"""Regression tests for external H1 active-support gating."""
from __future__ import annotations

import numpy as np

from scripts.run_external_suite import _h1_smoke


class _H1SupportDouble:
    n_agents = 3

    def __init__(self, active_after=None):
        self.t = 0
        self.active_after = active_after

    def get_action_dim(self):
        return 3

    def reset(self, seed=None):
        self.t = 0
        return [np.zeros(2, dtype=np.float32) for _ in range(self.n_agents)]

    def valid_action_mask(self, agent):
        return np.asarray([True, True, True], dtype=bool)

    def fixed_continuation_policy(self, agent):
        return 0

    def oracle_lag_response(
        self, ego_id, agent_j, intervention_action, horizon=2, **kwargs,
    ):
        active = bool(
            self.active_after is not None
            and self.t >= int(self.active_after)
            and int(ego_id) == 0
            and int(agent_j) == 1
            and int(intervention_action) != 0
        )
        per_lag = np.zeros(int(horizon), dtype=np.float64)
        if active:
            per_lag[0] = 0.25
        return {
            "per_lag_response": per_lag,
            "discounted_response": float(per_lag[0]),
            "response_mass": float(np.abs(per_lag).sum()),
        }

    def step(self, actions):
        self.t += 1
        obs = [np.asarray([i, self.t], dtype=np.float32) for i in range(self.n_agents)]
        rewards = [0.0] * self.n_agents
        return obs, rewards, False, {}


def test_h1_support_gate_does_not_promote_all_zero_oracle():
    result = _h1_smoke(_H1SupportDouble(active_after=None), max_states=4)
    assert result["interface_ready"] is True
    assert result["signal_ready"] is False
    assert result["valid_interventions"] > 0
    assert result["active_interventions"] == 0
    assert result["max_response_mass"] == 0.0


def test_h1_support_gate_scans_beyond_reset_state():
    result = _h1_smoke(_H1SupportDouble(active_after=2), max_states=5)
    assert result["states_tested"] >= 3
    assert result["signal_ready"] is True
    assert result["active_interventions"] > 0
    assert result["max_response_mass"] > 0.0
