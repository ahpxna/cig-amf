"""Environment-facing contract for domain-independent CIG-AMF features.

The learning code consumes this interface instead of reaching into a
simulator's private geometry. External environments can provide an adapter
through ``env.causal_adapter`` or register a wrapper at construction time.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np


PUBLIC_ROLES = (
    "collector", "gatekeeper", "relay", "blocker", "controller", "drifter"
)

TINY_PUBLIC_ROLES = ("hauler", "processor", "dispatcher", "spoiler")


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

    def candidate_neighbors(self, ego: int, max_degree: Optional[int]): ...

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
        action = int(action)
        if action < 0 or action >= self.max_action_dim:
            raise ValueError("neighbour action is outside the adapter action space")
        action_onehot = np.zeros((self.max_action_dim,), dtype=np.float32)
        action_onehot[action] = 1.0
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

    def candidate_neighbors(self, ego, max_degree):
        """Policy-independent bounded candidates for Eq. (28)."""
        ids = [j for j in range(self.n_agents) if int(j) != int(ego)]
        ids.sort(key=lambda j: (-self.weak_prior_score(ego, j), int(j)))
        return ids[: int(max_degree)]

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


class TinyOracleResourceFlowAdapter:
    """Causal feature contract for :class:`TinyOracleResourceFlowV1`.

    The tiny oracle environment has a resource-flow ontology rather than the
    OmniArena role and mechanism ontology.  This adapter deliberately exposes
    only public, pre-treatment target descriptors.  In particular it never
    reads hidden station degradation, hidden lane priorities, or any oracle
    influence label.  Target state is retained in ``x_ij`` because the
    leave-one-out context intentionally excludes the intervened neighbour.
    """

    _GEOMETRY_DIM = 5
    _ROLE_DIM = len(TINY_PUBLIC_ROLES)
    _CAPABILITY_DIM = 5  # four public role capabilities plus public efficacy
    _TARGET_STATE_DIM = 11

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
        if self._action_dim is not None:
            return int(self._action_dim)
        raise AttributeError("tiny adapter requires get_action_dim()")

    @property
    def obs_dim(self):
        return int(self.env.get_obs_dim())

    @property
    def pair_feature_dim(self):
        return (
            self._GEOMETRY_DIM + self._ROLE_DIM + self._CAPABILITY_DIM
            + self._TARGET_STATE_DIM
        )

    @property
    def relation_feature_dim(self):
        return self.pair_feature_dim

    @property
    def context_item_dim(self):
        # The pair descriptor, executed target action, and the target's public
        # resource-flow state form one observable set item.
        return self.pair_feature_dim + self.max_action_dim

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def observation(self, observations, agent):
        return np.asarray(
            self.env.get_obs_of_ego(observations, int(agent)), dtype=np.float32
        )

    def valid_action_mask(self, agent):
        del agent
        return np.ones((self.max_action_dim,), dtype=bool)

    @staticmethod
    def _visible(snapshot, ego, target):
        positions = snapshot["positions"]
        pi, pj = positions[int(ego)], positions[int(target)]
        radius = int(snapshot["obs_radius"])
        if max(abs(int(pi[0]) - int(pj[0])), abs(int(pi[1]) - int(pj[1]))) > radius:
            return False
        # Match TinyOracleResourceFlowV1's public occlusion rule without
        # peeking at latent task state.
        hidden = tuple(pj) in snapshot["occluders"] or tuple(pi) in snapshot["occluders"]
        vision = float(snapshot["agent_cap"][int(ego)][3])
        return not (hidden and vision < 0.95)

    @classmethod
    def _pair_features_from_tables(cls, snapshot, ego, target):
        ego, target = int(ego), int(target)
        grid = float(max(1, int(snapshot["grid_size"])))
        pi, pj = snapshot["positions"][ego], snapshot["positions"][target]
        zi, zj = int(snapshot["agent_zone"][ego]), int(snapshot["agent_zone"][target])
        visible = cls._visible(snapshot, ego, target)
        geometry = np.asarray([
            (float(pj[0]) - float(pi[0])) / grid if visible else 0.0,
            (float(pj[1]) - float(pi[1])) / grid if visible else 0.0,
            (abs(float(pj[0]) - float(pi[0])) + abs(float(pj[1]) - float(pi[1]))) / grid if visible else 0.0,
            float(zi == zj),
            (zj - zi) / float(max(1, int(snapshot["n_zones"]) - 1)),
        ], dtype=np.float32)
        role = np.zeros((cls._ROLE_DIM,), dtype=np.float32)
        try:
            role[TINY_PUBLIC_ROLES.index(str(snapshot["agent_role"][target]))] = 1.0
        except (ValueError, IndexError, KeyError, TypeError):
            pass
        cap = np.asarray(snapshot["agent_cap"][target], dtype=np.float32).reshape(-1)
        efficacy = float(snapshot["agent_structural_efficacy"][target])
        capability = np.zeros((cls._CAPABILITY_DIM,), dtype=np.float32)
        capability[:min(4, cap.size)] = cap[:4]
        capability[4] = efficacy
        # Dynamic target state is visible-only except public zone aggregates.
        zone_state = snapshot["zone_state"][zj]
        known_controller = float(snapshot["known_lane_controller"][ego][zj] == target)
        dynamic = np.asarray([
            float(visible),
            float(snapshot["inventory_type"][target]) / 2.0 if visible else 0.0,
            float(snapshot["inventory_qty"][target]) if visible else 0.0,
            float(snapshot["last_actions"][target]) / max(1.0, snapshot["action_dim"] - 1.0) if visible else 0.0,
            float(snapshot["last_signals"][target]) / 2.0 if visible else 0.0,
            float(zone_state[0]),  # lane open is public in Tiny observations
            float(zone_state[1]),  # normalized buffer stock
            float(zone_state[2]),  # normalized station progress
            float(zone_state[3]),  # station occupied
            float(zone_state[4]),  # normalized queue length
            known_controller,
        ], dtype=np.float32)
        return np.concatenate((geometry, role, capability, dynamic)).astype(np.float32)

    def pair_features(self, ego, target):
        return self._pair_features_from_tables(self.feature_snapshot(), ego, target)

    def pair_features_from_snapshot(self, snapshot, ego, target):
        return self._pair_features_from_tables(snapshot, ego, target)

    def relation_features(self, ego, target):
        return self.pair_features(ego, target)

    def compact_relation_features(self, ego, target, width):
        raw = self.pair_features(ego, target)
        out = np.zeros((int(width),), dtype=np.float32)
        out[:min(out.size, raw.size)] = raw[:min(out.size, raw.size)]
        return out

    def neighbour_features(self, ego, neighbour, action):
        action = int(action)
        if action < 0 or action >= self.max_action_dim:
            raise ValueError("neighbour action is outside the adapter action space")
        onehot = np.zeros((self.max_action_dim,), dtype=np.float32)
        onehot[action] = 1.0
        return np.concatenate((self.pair_features(ego, neighbour), onehot)).astype(np.float32)

    def context_key(self, ego, neighbour):
        pair = self.pair_features(ego, neighbour)
        return (
            int(pair[3] > 0.5),
            int(np.clip(pair[2] * self.env.grid_size, 0, 4)),
            str(self.env.agent_role[int(neighbour)]),
            int(pair[-11] > 0.5),
        )

    def weak_prior_score(self, ego, neighbour):
        pair = self.pair_features(ego, neighbour)
        return float(1.0 / (1.0 + pair[2] * self.env.grid_size) + 0.2 * pair[3])

    def candidate_neighbors(self, ego, max_degree):
        ids = [j for j in range(self.n_agents) if int(j) != int(ego)]
        ids.sort(key=lambda j: (-self.weak_prior_score(ego, j), int(j)))
        return ids[: int(max_degree)]

    def feature_snapshot(self):
        zone_state = []
        known = []
        for ego in range(self.n_agents):
            ego_known = []
            for zone in range(int(self.env.n_zones)):
                value = self.env.inspect_memory[ego].get(f"lane_controller_zone_{zone}")
                ego_known.append(
                    -1 if value is None else int(round(value * max(1, self.n_agents - 1)))
                )
            known.append(ego_known)
        for zone in range(int(self.env.n_zones)):
            buffer_node = self.env._get_zone_buffer(zone)
            lane_node = self.env._get_zone_lane(zone)
            station_node = self.env._get_zone_station(zone)
            zone_state.append([
                float(self.env.lane_open[lane_node.node_id]),
                float(self.env.buffer_stock[buffer_node.node_id]) / max(1.0, buffer_node.capacity),
                float(self.env.station_progress[station_node.node_id]) / 2.0,
                float(self.env.station_active_agent[station_node.node_id] != -1),
                float(len(self.env.station_queue[station_node.node_id])) / max(1.0, station_node.queue_limit),
            ])
        return {
            "positions": [list(value) for value in self.env.positions],
            "agent_zone": [int(value) for value in self.env.agent_zone],
            "agent_role": [str(value) for value in self.env.agent_role],
            "agent_cap": [np.asarray(value, dtype=np.float32).tolist() for value in self.env.agent_cap],
            "agent_structural_efficacy": [float(value) for value in self.env.agent_structural_efficacy],
            "inventory_type": [int(value) for value in self.env.inventory_type],
            "inventory_qty": [int(value) for value in self.env.inventory_qty],
            "last_actions": [int(value) for value in self.env.last_actions],
            "last_signals": [int(value) for value in self.env.last_signals],
            "zone_state": zone_state,
            "known_lane_controller": known,
            "grid_size": int(self.env.grid_size),
            "n_zones": int(self.env.n_zones),
            "obs_radius": int(self.env.obs_radius),
            "occluders": [tuple(value) for value in self.env.occluders],
            "action_dim": float(self.max_action_dim),
        }

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
    # Callers are allowed to pass an adapter directly.  The previous resolver
    # treated that object as though it were the raw environment and rejected
    # it unless it happened to expose OmniArena internals.
    if isinstance(env, CausalMultiAgentEnvAdapter):
        return env
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


def bounded_candidate_neighbors(
    adapter: CausalMultiAgentEnvAdapter,
    ego: int,
    max_degree: Optional[int],
) -> list[int]:
    """Return the authoritative policy-independent measured-neighbour set.

    Paper A Eq. (28) and Paper B Eq. (28) require this set to be constructed
    before causal scoring and independently of learned core allocation. A
    ``None`` bound deliberately preserves unrestricted all-pairs measurement;
    it must not be reported as a population-linear configuration.
    """
    ego = int(ego)
    n_agents = int(adapter.n_agents)
    if max_degree is None:
        return [j for j in range(n_agents) if j != ego]
    d_max = int(max_degree)
    if d_max <= 0:
        raise ValueError("candidate_max_degree must be positive or None")
    provider = getattr(adapter, "candidate_neighbors", None)
    if callable(provider):
        supplied = [int(j) for j in provider(ego, d_max)]
        ids = []
        seen = set()
        for j in supplied:
            if j == ego or j < 0 or j >= n_agents or j in seen:
                continue
            ids.append(j)
            seen.add(j)
        if len(ids) > d_max:
            raise ValueError("adapter candidate_neighbors exceeded d_max")
        return ids
    candidates = [j for j in range(n_agents) if j != ego]
    candidates.sort(
        key=lambda j: (-float(adapter.weak_prior_score(ego, j)), int(j))
    )
    return candidates[:d_max]
