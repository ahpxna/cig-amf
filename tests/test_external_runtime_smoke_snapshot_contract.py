"""Regression coverage for opaque external clone/restore snapshots."""
from __future__ import annotations

import numpy as np

from scripts.run_external_suite import _runtime_smoke


class _NonDeepcopyableSnapshot:
    """Snapshot token that deliberately rejects generic Python deepcopy."""

    def __init__(self, step_count):
        self.step_count = int(step_count)

    def __deepcopy__(self, memo):
        raise TypeError("snapshot is adapter-owned and must remain opaque")


class _OpaqueSnapshotEnv:
    n_agents = 2

    def __init__(self):
        self.step_count = 0

    def get_action_dim(self):
        return 3

    def get_obs_dim(self):
        return 2

    def reset(self, seed=None):
        del seed
        self.step_count = 0
        return self._get_obs_all()

    def _get_obs_all(self):
        return [
            np.asarray([float(agent), float(self.step_count)], dtype=np.float32)
            for agent in range(self.n_agents)
        ]

    def valid_action_mask(self, agent):
        del agent
        return np.asarray([True, True, True], dtype=bool)

    def clone_state(self):
        return _NonDeepcopyableSnapshot(self.step_count)

    def restore_state(self, state):
        if not isinstance(state, _NonDeepcopyableSnapshot):
            raise TypeError("restore_state requires the adapter snapshot token")
        self.step_count = int(state.step_count)

    def step(self, actions):
        assert len(actions) == self.n_agents
        self.step_count += 1
        return self._get_obs_all(), [1.0, 1.0], False, {}


def test_runtime_smoke_treats_clone_state_as_opaque_adapter_snapshot():
    env = _OpaqueSnapshotEnv()

    result = _runtime_smoke(env)

    assert env.step_count == 0
    assert result["n_agents"] == 2
    assert result["action_dim"] == 3
    assert result["obs_dim"] == 2
    assert result["reward_mean"] == 1.0
