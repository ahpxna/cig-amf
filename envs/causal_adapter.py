"""Environment-facing contract for domain-independent CIG-AMF features.

The learning code consumes this interface instead of reaching into a
simulator's private geometry. External environments can provide an adapter
through ``env.causal_adapter`` or register a wrapper at construction time.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


PUBLIC_ROLES = (
    "collector", "gatekeeper", "relay", "blocker", "controller", "drifter"
)


@runtime_checkable
class CausalMultiAgentEnvAdapter(Protocol):
    @property
    def n_agents(self) -> int: ...

    @property
    def max_action_dim(self) -> int: ...

    @property
    def obs_dim(self) -> int: ...

    @property
    def pair_feature_dim(self) -> int: ...

    @property
    def relation_feature_dim(self) -> int: ...

    @property
    def context_item_dim(self) -> int: ...

    def observation(self, observations, agent: int) -> np.ndarray: ...

    def reset(self): ...

    def step(self, actions): ...

    def valid_action_mask(self, agent: int) -> np.ndarray: ...

    def pair_features(self, ego: int, target: int) -> np.ndarray: ...

    def relation_features(self, ego: int, target: int) -> np.ndarray: ...

    def compact_relation_features(
        self, ego: int, target: int, width: int
    ) -> np.ndarray: ...

    def neighbour_features(
        self, ego: int, neighbour: int, action: int
    ) -> np.ndarray: ...

    def context_key(self, ego: int, neighbour: int): ...

    def weak_prior_score(self, ego: int, neighbour: int) -> float: ...

    def feature_snapshot(self) -> dict: ...

    def pair_features_from_snapshot(
        self, snapshot: dict, ego: int, target: int
    ) -> np.ndarray: ...

    def clone_state(self): ...

    def restore_state(self, state): ...

    def fixed_continuation_policy(self, agent: int) -> int: ...


class OmniArenaAdapter:
    """Expose OmniArena through the generic causal feature contract."""

    def __init__(self, env, action_dim=None):
        self.env = env
        self._action_dim = None if action_dim is None else int(action_dim)

    @property
    def n_agents(self):
        return int(self.env.n_agents)

    @property
    def max_action_dim(self):
        getter = getattr(self.env, "get_action_dim", None)
        if callable(getter):
            return int(getter())
        if hasattr(self.env, "N_ACTIONS"):
            return int(self.env.N_ACTIONS)
        if self._action_dim is not None:
            return int(self._action_dim)
        raise AttributeError(
            "adapter requires get_action_dim(), N_ACTIONS, or action_dim"
        )

    @property
    def obs_dim(self):
        return int(self.env.get_obs_dim())

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    @property
    def pair_feature_dim(self):
        return 5 + len(PUBLIC_ROLES)

    @property
    def relation_feature_dim(self):
        return self.pair_feature_dim

    @property
    def context_item_dim(self):
        # Pair geometry/capability, executed action, and four observable
        # mechanism-state indicators.
        return self.pair_feature_dim + self.max_action_dim + 4

    def observation(self, observations, agent):
        return np.asarray(
            self.env.get_obs_of_ego(observations, int(agent)), dtype=np.float32
        )

    def valid_action_mask(self, agent):
        provider = getattr(self.env, "valid_action_mask", None)
        if callable(provider):
            mask = np.asarray(provider(int(agent)), dtype=bool).reshape(-1)
        else:
            mask = np.ones((self.max_action_dim,), dtype=bool)
        if mask.shape != (self.max_action_dim,) or not np.any(mask):
            raise ValueError(
                "valid_action_mask must contain at least one valid action and "
                f"have shape {(self.max_action_dim,)}, got {mask.shape}"
            )
        return mask

    @staticmethod
    def _pair_features_from_tables(
        positions, agent_zone, grid_size, n_zones, ego, target, agent_role=None
    ):
        grid = float(max(1, int(grid_size)))
        pi = positions[int(ego)]
        pj = positions[int(target)]
        drow = (float(pj[0]) - float(pi[0])) / grid
        dcol = (float(pj[1]) - float(pi[1])) / grid
        distance = (
            abs(float(pj[0]) - float(pi[0]))
            + abs(float(pj[1]) - float(pi[1]))
        ) / grid
        zi = int(agent_zone[int(ego)])
        zj = int(agent_zone[int(target)])
        role_onehot = np.zeros((len(PUBLIC_ROLES),), dtype=np.float32)
        if agent_role is not None:
            try:
                role_onehot[PUBLIC_ROLES.index(str(agent_role[int(target)]))] = 1.0
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        return np.concatenate([
            np.asarray([
                drow,
                dcol,
                distance,
                float(zi == zj),
                (zj - zi) / float(max(1, int(n_zones) - 1)),
            ], dtype=np.float32),
            role_onehot,
        ])

    def pair_features(self, ego, target):
        return self._pair_features_from_tables(
            self.env.positions,
            self.env.agent_zone,
            self.env.grid_size,
            self.env.n_zones,
            ego,
            target,
            getattr(self.env, "agent_role", None),
        )

    def relation_features(self, ego, target):
        """Observable domain relation xi_ij; pair_features is its Omni view."""
        return self.pair_features(ego, target)

    def compact_relation_features(self, ego, target, width):
        """Adapter-owned fixed-width view for memory and belief encoders."""
        raw = self.pair_features(ego, target)
        # Keep the historical Omni information budget, but model modules do
        # not know that these channels originated from grid geometry.
        selected = np.asarray([raw[0], raw[1], raw[4], raw[2]], dtype=np.float32)
        out = np.zeros((int(width),), dtype=np.float32)
        out[:min(out.size, selected.size)] = selected[:min(out.size, selected.size)]
        return out

    def _mechanism_features(self, neighbour):
        neighbour = int(neighbour)
        zone = int(self.env.agent_zone[neighbour])
        return np.asarray([
            float(bool(getattr(self.env, "gate_open", {}).get(zone, False))),
            float(not bool(
                getattr(self.env, "resource_available", {}).get(zone, True)
            )),
            float(bool(getattr(self.env, "carrying", {}).get(neighbour, False))),
            float(str(getattr(self.env, "active_lane", {}).get(zone, "A")) == "B"),
        ], dtype=np.float32)

    def neighbour_features(self, ego, neighbour, action):
        action_onehot = np.zeros((self.max_action_dim,), dtype=np.float32)
        action_onehot[int(np.clip(action, 0, self.max_action_dim - 1))] = 1.0
        return np.concatenate([
            self.pair_features(ego, neighbour),
            action_onehot,
            self._mechanism_features(neighbour),
        ]).astype(np.float32)

    def context_key(self, ego, neighbour):
        pair = self.pair_features(ego, neighbour)
        distance = float(pair[2])
        distance_bin = 0 if distance <= 2.0 / max(1, self.env.grid_size) else (
            1 if distance <= 5.0 / max(1, self.env.grid_size) else 2
        )
        role = "unknown"
        try:
            role = str(self.env.agent_role[int(neighbour)])
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        mechanism = self._mechanism_features(neighbour).astype(np.int64)
        mechanism_bits = sum(int(value) << index for index, value in enumerate(mechanism))
        return (
            int(pair[3] > 0.5),
            int(distance_bin),
            int(self.env.agent_zone[int(ego)]),
            role,
            int(mechanism_bits),
        )

    def weak_prior_score(self, ego, neighbour):
        pair = self.pair_features(ego, neighbour)
        raw_distance = float(pair[2]) * float(max(1, int(self.env.grid_size)))
        role_bonus = {
            "hauler": 0.20,
            "processor": 0.26,
            "dispatcher": 0.24,
            "sweeper": 0.10,
            "spoiler": 0.08,
        }
        role = None
        try:
            role = str(self.env.agent_role[int(neighbour)])
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        return float(
            1.0 / (1.0 + raw_distance)
            + (0.25 if float(pair[3]) > 0.5 else 0.0)
            + role_bonus.get(role, 0.0)
        )

    def feature_snapshot(self):
        return {
            "positions": [
                list(self.env.positions[index]) for index in range(self.n_agents)
            ],
            "agent_zone": [
                int(self.env.agent_zone[index]) for index in range(self.n_agents)
            ],
            "agent_role": (
                [str(self.env.agent_role[index]) for index in range(self.n_agents)]
                if hasattr(self.env, "agent_role") else None
            ),
            "grid_size": int(self.env.grid_size),
            "n_zones": int(self.env.n_zones),
        }

    def pair_features_from_snapshot(self, snapshot, ego, target):
        return self._pair_features_from_tables(
            snapshot["positions"], snapshot["agent_zone"],
            snapshot["grid_size"], snapshot["n_zones"], ego, target,
            snapshot.get("agent_role"),
        )

    def clone_state(self):
        return self.env.clone_state()

    def restore_state(self, state):
        return self.env.restore_state(state)

    def fixed_continuation_policy(self, agent):
        return int(self.env.scripted_policy(int(agent)))


def resolve_env_adapter(
    env, adapter: Optional[CausalMultiAgentEnvAdapter] = None, action_dim=None
):
    if adapter is not None:
        return adapter
    provided = getattr(env, "causal_adapter", None)
    if callable(provided):
        provided = provided()
    if provided is not None:
        return provided
    required = ("positions", "agent_zone", "grid_size", "n_zones")
    if all(hasattr(env, name) for name in required):
        return OmniArenaAdapter(env, action_dim=action_dim)
    raise TypeError(
        "environment must expose causal_adapter or be wrapped by a "
        "CausalMultiAgentEnvAdapter"
    )


def compact_relation_features(adapter, ego, target, width):
    """Return an opaque fixed-width relation representation from an adapter.

    This is the only bridge from variable-size domain relation vectors to
    legacy fixed-width memory encoders. New adapters may provide a learned or
    domain-specific projection; the fallback is shape-only pad/truncate and
    never assigns semantics to positional channels in a model module.
    """
    provider = getattr(adapter, "compact_relation_features", None)
    if callable(provider):
        value = provider(int(ego), int(target), int(width))
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape != (int(width),):
            raise ValueError(
                "compact_relation_features must return shape "
                f"{(int(width),)}, got {value.shape}"
            )
        return value
    raw = np.asarray(adapter.relation_features(int(ego), int(target)), dtype=np.float32)
    out = np.zeros((int(width),), dtype=np.float32)
    out[:min(out.size, raw.size)] = raw[:min(out.size, raw.size)]
    return out
