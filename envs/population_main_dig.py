import copy
import numpy as np


class PopulationMainDIG:
    """Dynamic Influence Gridworld with full population, multiple ego agents.

Roles:
- collector: agent that collects reward within its zone
- gatekeeper: opens the gate
- relay: provides indirect support
- blocker: hinders the collector
- drifter: causes mild disturbance

Each ego has:
- private reward
- private core
- private influence"""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    OPEN = 5
    N_ACTIONS = 6

    ROLE_COLLECTOR = "collector"
    ROLE_GATEKEEPER = "gatekeeper"
    ROLE_RELAY = "relay"
    ROLE_BLOCKER = "blocker"
    ROLE_DRIFTER = "drifter"

    def __init__(
        self,
        n_agents=24,
        grid_size=16,
        n_zones=4,
        obs_radius=4,
        max_steps=60,
        phase_length=40,
        mode="behavioral_drift",
        seed=42,
    ):
        assert n_agents >= 4 * n_zones, "Cần ít nhất 4 agents cho mỗi zone."
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_zones = n_zones
        self.obs_radius = obs_radius
        self.max_steps = max_steps
        self.phase_length = phase_length
        self.mode = mode
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.supported_egos = list(range(self.n_agents))
        self.current_phase = 0
        self.episode_count = 0

        self._init_zone_layout()
        self._init_population_roles()
        self.reset()

    # ============================================================
    # Layout
    # ============================================================

    def _init_zone_layout(self):
        self.zone_resource = {}
        self.zone_gate = {}
        self.zone_relay = {}
        self.zone_centers = {}

        rows = max(2, int(np.sqrt(self.n_zones)))
        cols = int(np.ceil(self.n_zones / rows))
        cell_h = self.grid_size // rows
        cell_w = self.grid_size // cols

        zone = 0
        for rr in range(rows):
            for cc in range(cols):
                if zone >= self.n_zones:
                    break
                r0 = rr * cell_h
                c0 = cc * cell_w
                r1 = min(self.grid_size - 1, r0 + cell_h - 1)
                c1 = min(self.grid_size - 1, c0 + cell_w - 1)

                center = ((r0 + r1) // 2, (c0 + c1) // 2)
                self.zone_centers[zone] = center
                self.zone_gate[zone] = (max(0, center[0] - 1), center[1])
                self.zone_resource[zone] = (min(self.grid_size - 1, center[0] + 1), center[1])
                self.zone_relay[zone] = (center[0], max(0, center[1] - 1))
                zone += 1

    # ============================================================
    # Roles and graph truth
    # ============================================================

    def _init_population_roles(self):
        self.agent_role = {}
        self.agent_zone = {}
        self.zone_role_agents = {}

        idx = 0
        for z in range(self.n_zones):
            collector = idx
            gatekeeper = idx + 1
            relay = idx + 2
            blocker = idx + 3
            idx += 4

            self.zone_role_agents[z] = {
                self.ROLE_COLLECTOR: collector,
                self.ROLE_GATEKEEPER: gatekeeper,
                self.ROLE_RELAY: relay,
                self.ROLE_BLOCKER: blocker,
            }

            self.agent_role[collector] = self.ROLE_COLLECTOR
            self.agent_role[gatekeeper] = self.ROLE_GATEKEEPER
            self.agent_role[relay] = self.ROLE_RELAY
            self.agent_role[blocker] = self.ROLE_BLOCKER

            self.agent_zone[collector] = z
            self.agent_zone[gatekeeper] = z
            self.agent_zone[relay] = z
            self.agent_zone[blocker] = z

        for a in range(idx, self.n_agents):
            self.agent_role[a] = self.ROLE_DRIFTER
            self.agent_zone[a] = self.rng.randint(0, self.n_zones)

        self._refresh_gt_graph()

    def _refresh_gt_graph(self):
        self.gt_core_by_ego = {}
        self.gt_influence_by_ego = {}

        for ego in range(self.n_agents):
            role = self.agent_role[ego]
            zone = self.agent_zone[ego]
            role_agents = self.zone_role_agents[zone]

            core = set()
            influence = {}

            if role == self.ROLE_COLLECTOR:
                core = {
                    role_agents[self.ROLE_GATEKEEPER],
                    role_agents[self.ROLE_RELAY],
                    role_agents[self.ROLE_BLOCKER],
                }
                influence = {
                    role_agents[self.ROLE_GATEKEEPER]: 1.0,
                    role_agents[self.ROLE_RELAY]: 0.55,
                    role_agents[self.ROLE_BLOCKER]: -0.80,
                }

            elif role == self.ROLE_GATEKEEPER:
                core = {
                    role_agents[self.ROLE_COLLECTOR],
                    role_agents[self.ROLE_RELAY],
                }
                influence = {
                    role_agents[self.ROLE_COLLECTOR]: 0.60,
                    role_agents[self.ROLE_RELAY]: 0.35,
                }

            elif role == self.ROLE_RELAY:
                core = {
                    role_agents[self.ROLE_COLLECTOR],
                    role_agents[self.ROLE_GATEKEEPER],
                }
                influence = {
                    role_agents[self.ROLE_COLLECTOR]: 0.40,
                    role_agents[self.ROLE_GATEKEEPER]: 0.65,
                }

            elif role == self.ROLE_BLOCKER:
                core = {role_agents[self.ROLE_COLLECTOR]}
                influence = {
                    role_agents[self.ROLE_COLLECTOR]: -0.70,
                }

            else:
                collector = role_agents[self.ROLE_COLLECTOR]
                gatekeeper = role_agents[self.ROLE_GATEKEEPER]
                core = {collector, gatekeeper}
                influence = {
                    collector: 0.20,
                    gatekeeper: 0.15,
                }

            full = {}
            for j in range(self.n_agents):
                if j == ego:
                    continue
                full[j] = float(influence.get(j, 0.0))

            self.gt_core_by_ego[ego] = set(core)
            self.gt_influence_by_ego[ego] = full

    # ============================================================
    # Structural shift
    # ============================================================

    def _maybe_structural_shift(self):
        if self.mode != "structural_shift":
            return

        if self.episode_count > 0 and self.episode_count % self.phase_length == 0:
            self.current_phase += 1

            # switch gatekeeper between zones
            for z in range(self.n_zones):
                nxt = (z + 1) % self.n_zones
                self.zone_role_agents[z][self.ROLE_GATEKEEPER], self.zone_role_agents[nxt][self.ROLE_GATEKEEPER] = \
                    self.zone_role_agents[nxt][self.ROLE_GATEKEEPER], self.zone_role_agents[z][self.ROLE_GATEKEEPER]

            # sync role + zone
            for z in range(self.n_zones):
                for role_name, aid in self.zone_role_agents[z].items():
                    self.agent_role[aid] = role_name
                    self.agent_zone[aid] = z

            self._refresh_gt_graph()

    # ============================================================
    # API
    # ============================================================

    def get_supported_egos(self):
        return list(self.supported_egos)

    def get_gt_core_for_ego(self, ego_id):
        return set(self.gt_core_by_ego[ego_id])

    def get_gt_influence_for_ego(self, ego_id):
        return dict(self.gt_influence_by_ego[ego_id])

    def get_obs_of_ego(self, obs_all, ego_id):
        return obs_all[ego_id]

    def get_reward_of_ego(self, rewards, ego_id):
        return float(rewards[ego_id])

    def get_obs_dim(self):
        return len(self._get_obs(0))

    def get_action_dim(self):
        return self.N_ACTIONS

    def clone_state(self):
        return {
            "positions": copy.deepcopy(self.positions),
            "gate_open": copy.deepcopy(self.gate_open),
            "resource_available": copy.deepcopy(self.resource_available),
            "last_actions": copy.deepcopy(self.last_actions),
            "t": self.t,
            "done": self.done,
            "episode_count": self.episode_count,
            "current_phase": self.current_phase,
            "rng_state": self.rng.get_state(),
            "zone_role_agents": copy.deepcopy(self.zone_role_agents),
            "agent_role": copy.deepcopy(self.agent_role),
            "agent_zone": copy.deepcopy(self.agent_zone),
            "gt_core_by_ego": copy.deepcopy(self.gt_core_by_ego),
            "gt_influence_by_ego": copy.deepcopy(self.gt_influence_by_ego),
        }

    def restore_state(self, state):
        self.positions = copy.deepcopy(state["positions"])
        self.gate_open = copy.deepcopy(state["gate_open"])
        self.resource_available = copy.deepcopy(state["resource_available"])
        self.last_actions = copy.deepcopy(state["last_actions"])
        self.t = state["t"]
        self.done = state["done"]
        self.episode_count = state["episode_count"]
        self.current_phase = state["current_phase"]
        self.rng.set_state(state["rng_state"])
        self.zone_role_agents = copy.deepcopy(state["zone_role_agents"])
        self.agent_role = copy.deepcopy(state["agent_role"])
        self.agent_zone = copy.deepcopy(state["agent_zone"])
        self.gt_core_by_ego = copy.deepcopy(state["gt_core_by_ego"])
        self.gt_influence_by_ego = copy.deepcopy(state["gt_influence_by_ego"])

    def reset(self):
        self._maybe_structural_shift()

        self.positions = {}
        for a in range(self.n_agents):
            z = self.agent_zone[a]
            center = self.zone_centers[z]
            rr = self.rng.randint(max(0, center[0] - 2), min(self.grid_size, center[0] + 3))
            cc = self.rng.randint(max(0, center[1] - 2), min(self.grid_size, center[1] + 3))
            self.positions[a] = [rr, cc]

        self.gate_open = {z: False for z in range(self.n_zones)}
        self.resource_available = {z: True for z in range(self.n_zones)}
        self.last_actions = {a: self.STAY for a in range(self.n_agents)}
        self.t = 0
        self.done = False
        self.episode_count += 1
        return self._get_obs_all()

    # ============================================================
    # Helpers
    # ============================================================

    def _clip(self, x):
        return max(0, min(self.grid_size - 1, x))

    def _move(self, pos, action):
        r, c = pos
        if action == self.UP:
            r -= 1
        elif action == self.DOWN:
            r += 1
        elif action == self.LEFT:
            c -= 1
        elif action == self.RIGHT:
            c += 1
        return [self._clip(r), self._clip(c)]

    def _dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbor_mode(self, agent_id):
        if self.mode == "behavioral_drift":
            phase = (self.episode_count // self.phase_length) % 4
            if phase == 0:
                return "cooperative"
            if phase == 1:
                return "delayed"
            if phase == 2:
                return "zigzag"
            return "lazy"
        return "cooperative"

    # ============================================================
    # Scripted policies for oracle-free environment dynamics
    # ============================================================

    def scripted_policy(self, agent_id):
        role = self.agent_role[agent_id]
        zone = self.agent_zone[agent_id]
        pos = self.positions[agent_id]

        gate = self.zone_gate[zone]
        res = self.zone_resource[zone]
        relay = self.zone_relay[zone]
        collector = self.zone_role_agents[zone][self.ROLE_COLLECTOR]

        def greedy(src, dst):
            sr, sc = src
            tr, tc = dst
            if sr < tr:
                return self.DOWN
            if sr > tr:
                return self.UP
            if sc < tc:
                return self.RIGHT
            if sc > tc:
                return self.LEFT
            return self.STAY

        mode = self._neighbor_mode(agent_id)

        if role == self.ROLE_COLLECTOR:
            return greedy(pos, res)

        if role == self.ROLE_GATEKEEPER:
            if tuple(pos) == gate:
                if mode == "cooperative":
                    return self.OPEN
                if mode == "delayed":
                    return self.OPEN if (self.t % 2 == 0) else self.STAY
                if mode == "zigzag":
                    return self.OPEN if (self.t % 3 != 1) else self.LEFT
                return self.OPEN if (self.t % 3 == 0) else self.STAY
            return greedy(pos, gate)

        if role == self.ROLE_RELAY:
            if mode == "zigzag" and self.t % 4 == 0:
                return self.LEFT
            return greedy(pos, relay)

        if role == self.ROLE_BLOCKER:
            collector_pos = self.positions[collector]
            return greedy(pos, collector_pos)

        if agent_id % 3 == 0:
            return self.STAY
        if agent_id % 3 == 1:
            return self.rng.randint(0, self.N_ACTIONS - 1)
        return self.STAY if (self.t % 2 == 0) else self.rng.randint(0, self.N_ACTIONS - 1)

    # ============================================================
    # Step
    # ============================================================

    def step(self, actions):
        if self.done:
            return self._get_obs_all(), [0.0] * self.n_agents, True, {}

        rewards = [-0.01 for _ in range(self.n_agents)]

        # update gate states before movement
        for z in range(self.n_zones):
            gk = self.zone_role_agents[z][self.ROLE_GATEKEEPER]
            gate_pos = self.zone_gate[z]
            self.gate_open[z] = (tuple(self.positions[gk]) == gate_pos and actions[gk] == self.OPEN)

        # movement
        proposed = {}
        for a in range(self.n_agents):
            if actions[a] == self.OPEN:
                proposed[a] = list(self.positions[a])
            else:
                proposed[a] = self._move(self.positions[a], actions[a])

        occupancy = {}
        for a in range(self.n_agents):
            key = tuple(proposed[a])
            occupancy[key] = occupancy.get(key, 0) + 1

        for a in range(self.n_agents):
            if occupancy[tuple(proposed[a])] == 1:
                self.positions[a] = proposed[a]
            else:
                rewards[a] -= 0.05

        self.last_actions = {a: int(actions[a]) for a in range(self.n_agents)}

        # rewards per ego
        for ego in range(self.n_agents):
            role = self.agent_role[ego]
            zone = self.agent_zone[ego]
            role_agents = self.zone_role_agents[zone]

            collector = role_agents[self.ROLE_COLLECTOR]
            gatekeeper = role_agents[self.ROLE_GATEKEEPER]
            relay = role_agents[self.ROLE_RELAY]
            blocker = role_agents[self.ROLE_BLOCKER]

            ego_pos = tuple(self.positions[ego])
            collector_pos = tuple(self.positions[collector])
            gatekeeper_pos = tuple(self.positions[gatekeeper])
            relay_pos = tuple(self.positions[relay])

            gate_pos = self.zone_gate[zone]
            res_pos = self.zone_resource[zone]
            relay_target = self.zone_relay[zone]

            if role == self.ROLE_COLLECTOR:
                min_dist = self._dist(self.positions[ego], res_pos)
                rewards[ego] += 0.02 / (min_dist + 1)

                if self.gate_open[zone] and self.resource_available[zone] and ego_pos == res_pos:
                    rewards[ego] += 1.0
                    self.resource_available[zone] = False

            if role == self.ROLE_GATEKEEPER:
                if gatekeeper_pos == gate_pos:
                    rewards[ego] += 0.12
                if self.gate_open[zone]:
                    rewards[ego] += 0.10

            if role == self.ROLE_RELAY:
                if relay_pos == relay_target:
                    rewards[ego] += 0.10
                if relay_pos == relay_target and self._dist(self.positions[gatekeeper], gate_pos) <= 1:
                    rewards[ego] += 0.08

            if role == self.ROLE_BLOCKER:
                d = self._dist(self.positions[ego], collector_pos)
                rewards[ego] += 0.08 / (d + 1)

            # interaction terms
            if ego == collector and gatekeeper_pos == gate_pos:
                rewards[ego] += 0.25

            if ego == collector and relay_pos == relay_target and self._dist(self.positions[gatekeeper], gate_pos) <= 1:
                rewards[ego] += 0.12

            if ego == collector and self._dist(self.positions[blocker], self.positions[collector]) <= 1:
                rewards[ego] -= 0.18

            if ego == gatekeeper and self._dist(self.positions[collector], res_pos) <= 1:
                rewards[ego] += 0.05

            if ego == gatekeeper and relay_pos == relay_target:
                rewards[ego] += 0.05

            # nuisance from drifters
            for j in range(self.n_agents):
                if j == ego:
                    continue
                if self.agent_role[j] == self.ROLE_DRIFTER and self._dist(self.positions[j], self.positions[ego]) <= 1:
                    rewards[ego] -= 0.01

        for z in range(self.n_zones):
            if (not self.resource_available[z]) and self.rng.rand() < 0.03:
                self.resource_available[z] = True

        self.t += 1
        self.done = (self.t >= self.max_steps)

        info = {
            "gt_core_by_ego": copy.deepcopy(self.gt_core_by_ego),
            "gt_influence_by_ego": copy.deepcopy(self.gt_influence_by_ego),
            "current_phase": self.current_phase,
            "mode": self.mode,
            "evaluation_scope": "population_wide_multi_ego",
        }

        return self._get_obs_all(), rewards, self.done, info

    # ============================================================
    # Observations
    # ============================================================

    def _get_obs(self, ego):
        own = self.positions[ego]
        zone = self.agent_zone[ego]
        role = self.agent_role[ego]

        role_to_id = {
            self.ROLE_COLLECTOR: 0,
            self.ROLE_GATEKEEPER: 1,
            self.ROLE_RELAY: 2,
            self.ROLE_BLOCKER: 3,
            self.ROLE_DRIFTER: 4,
        }

        obs = [
            own[0] / self.grid_size,
            own[1] / self.grid_size,
            zone / max(1, self.n_zones - 1),
            role_to_id[role] / 4.0,
        ]

        gate = self.zone_gate[zone]
        res = self.zone_resource[zone]
        relay = self.zone_relay[zone]

        obs.extend([
            (gate[0] - own[0]) / self.grid_size,
            (gate[1] - own[1]) / self.grid_size,
            (res[0] - own[0]) / self.grid_size,
            (res[1] - own[1]) / self.grid_size,
            (relay[0] - own[0]) / self.grid_size,
            (relay[1] - own[1]) / self.grid_size,
            float(self.gate_open[zone]),
            float(self.resource_available[zone]),
        ])

        neighbors = []
        for j in range(self.n_agents):
            if j == ego:
                continue
            if self._dist(self.positions[ego], self.positions[j]) <= self.obs_radius:
                pj = self.positions[j]
                neighbors.extend([
                    (pj[0] - own[0]) / self.grid_size,
                    (pj[1] - own[1]) / self.grid_size,
                    self.agent_zone[j] / max(1, self.n_zones - 1),
                    role_to_id[self.agent_role[j]] / 4.0,
                    self.last_actions[j] / max(1, self.N_ACTIONS - 1),
                ])

        max_nb = min(8, self.n_agents - 1)
        target_len = max_nb * 5
        while len(neighbors) < target_len:
            neighbors.extend([-1.0, -1.0, -1.0, -1.0, -1.0])

        obs.extend(neighbors[:target_len])
        return np.array(obs, dtype=np.float32)

    def _get_obs_all(self):
        return [self._get_obs(i) for i in range(self.n_agents)]
