import copy
import numpy as np


class TinyOracleDIG:
    """
    Tiny Oracle nhiều ego dùng cho calibration.

    Mỗi zone có:
    - 1 collector
    - 1 gatekeeper
    - 1 relay
    - 1 blocker

    Clone state được để đo effect can thiệp thật.
    """

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

    def __init__(self, grid_size=8, max_steps=24, causal_horizon=3, seed=42):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.causal_horizon = causal_horizon
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.n_zones = 2
        self.n_agents = 8
        self.supported_egos = list(range(self.n_agents))

        self.zone_gate = {0: (2, 2), 1: (2, 5)}
        self.zone_resource = {0: (4, 2), 1: (4, 5)}
        self.zone_relay = {0: (3, 1), 1: (3, 4)}

        self.zone_role_agents = {
            0: {
                self.ROLE_COLLECTOR: 0,
                self.ROLE_GATEKEEPER: 1,
                self.ROLE_RELAY: 2,
                self.ROLE_BLOCKER: 3,
            },
            1: {
                self.ROLE_COLLECTOR: 4,
                self.ROLE_GATEKEEPER: 5,
                self.ROLE_RELAY: 6,
                self.ROLE_BLOCKER: 7,
            }
        }

        self.agent_role = {}
        self.agent_zone = {}
        for z in range(self.n_zones):
            for role, aid in self.zone_role_agents[z].items():
                self.agent_role[aid] = role
                self.agent_zone[aid] = z

        self._refresh_gt_graph()
        self.reset()

    def _refresh_gt_graph(self):
        self.gt_core_by_ego = {}
        self.gt_influence_by_ego = {}

        for ego in range(self.n_agents):
            z = self.agent_zone[ego]
            role = self.agent_role[ego]
            role_agents = self.zone_role_agents[z]

            if role == self.ROLE_COLLECTOR:
                core = {
                    role_agents[self.ROLE_GATEKEEPER],
                    role_agents[self.ROLE_RELAY],
                    role_agents[self.ROLE_BLOCKER],
                }
                inf = {
                    role_agents[self.ROLE_GATEKEEPER]: 1.0,
                    role_agents[self.ROLE_RELAY]: 0.55,
                    role_agents[self.ROLE_BLOCKER]: -0.80,
                }
            elif role == self.ROLE_GATEKEEPER:
                core = {
                    role_agents[self.ROLE_COLLECTOR],
                    role_agents[self.ROLE_RELAY],
                }
                inf = {
                    role_agents[self.ROLE_COLLECTOR]: 0.60,
                    role_agents[self.ROLE_RELAY]: 0.35,
                }
            elif role == self.ROLE_RELAY:
                core = {
                    role_agents[self.ROLE_COLLECTOR],
                    role_agents[self.ROLE_GATEKEEPER],
                }
                inf = {
                    role_agents[self.ROLE_COLLECTOR]: 0.40,
                    role_agents[self.ROLE_GATEKEEPER]: 0.65,
                }
            else:
                core = {role_agents[self.ROLE_COLLECTOR]}
                inf = {role_agents[self.ROLE_COLLECTOR]: -0.70}

            full = {}
            for j in range(self.n_agents):
                if j == ego:
                    continue
                full[j] = float(inf.get(j, 0.0))

            self.gt_core_by_ego[ego] = set(core)
            self.gt_influence_by_ego[ego] = full

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
            "rng_state": self.rng.get_state(),
        }

    def restore_state(self, state):
        self.positions = copy.deepcopy(state["positions"])
        self.gate_open = copy.deepcopy(state["gate_open"])
        self.resource_available = copy.deepcopy(state["resource_available"])
        self.last_actions = copy.deepcopy(state["last_actions"])
        self.t = state["t"]
        self.done = state["done"]
        self.rng.set_state(state["rng_state"])

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

    def scripted_policy(self, agent_id):
        role = self.agent_role[agent_id]
        z = self.agent_zone[agent_id]
        pos = self.positions[agent_id]

        gate = self.zone_gate[z]
        res = self.zone_resource[z]
        relay = self.zone_relay[z]
        collector = self.zone_role_agents[z][self.ROLE_COLLECTOR]

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

        if role == self.ROLE_COLLECTOR:
            return greedy(pos, res)
        if role == self.ROLE_GATEKEEPER:
            if tuple(pos) == gate:
                return self.OPEN
            return greedy(pos, gate)
        if role == self.ROLE_RELAY:
            return greedy(pos, relay)
        return greedy(pos, self.positions[collector])

    def reset(self):
        self.positions = {
            0: [1, 2], 1: [0, 2], 2: [1, 1], 3: [0, 0],
            4: [1, 5], 5: [0, 5], 6: [1, 4], 7: [0, 7],
        }
        self.gate_open = {0: False, 1: False}
        self.resource_available = {0: True, 1: True}
        self.last_actions = {a: self.STAY for a in range(self.n_agents)}
        self.t = 0
        self.done = False
        return self._get_obs_all()

    def step(self, actions):
        if self.done:
            return self._get_obs_all(), [0.0] * self.n_agents, True, {}

        rewards = [-0.01 for _ in range(self.n_agents)]

        for z in range(self.n_zones):
            gk = self.zone_role_agents[z][self.ROLE_GATEKEEPER]
            self.gate_open[z] = (tuple(self.positions[gk]) == self.zone_gate[z] and actions[gk] == self.OPEN)

        proposed = {}
        for a in range(self.n_agents):
            if actions[a] == self.OPEN:
                proposed[a] = list(self.positions[a])
            else:
                proposed[a] = self._move(self.positions[a], actions[a])

        self.positions = proposed
        self.last_actions = {a: int(actions[a]) for a in range(self.n_agents)}

        for ego in range(self.n_agents):
            z = self.agent_zone[ego]
            role = self.agent_role[ego]
            role_agents = self.zone_role_agents[z]

            collector = role_agents[self.ROLE_COLLECTOR]
            gatekeeper = role_agents[self.ROLE_GATEKEEPER]
            relay = role_agents[self.ROLE_RELAY]
            blocker = role_agents[self.ROLE_BLOCKER]

            ego_pos = tuple(self.positions[ego])

            if role == self.ROLE_COLLECTOR:
                if self.gate_open[z] and self.resource_available[z] and ego_pos == self.zone_resource[z]:
                    rewards[ego] += 1.0
                    self.resource_available[z] = False
                if tuple(self.positions[gatekeeper]) == self.zone_gate[z]:
                    rewards[ego] += 0.25
                if tuple(self.positions[relay]) == self.zone_relay[z]:
                    rewards[ego] += 0.12
                if self._dist(self.positions[blocker], self.positions[collector]) <= 1:
                    rewards[ego] -= 0.18

            elif role == self.ROLE_GATEKEEPER:
                if tuple(self.positions[gatekeeper]) == self.zone_gate[z]:
                    rewards[ego] += 0.10

            elif role == self.ROLE_RELAY:
                if tuple(self.positions[relay]) == self.zone_relay[z]:
                    rewards[ego] += 0.08

            elif role == self.ROLE_BLOCKER:
                if self._dist(self.positions[blocker], self.positions[collector]) <= 1:
                    rewards[ego] += 0.10

        self.t += 1
        self.done = (self.t >= self.max_steps)

        info = {
            "gt_core_by_ego": copy.deepcopy(self.gt_core_by_ego),
            "gt_influence_by_ego": copy.deepcopy(self.gt_influence_by_ego),
            "evaluation_scope": "tiny_oracle_multi_ego",
        }

        return self._get_obs_all(), rewards, self.done, info

    def _get_obs(self, ego):
        own = self.positions[ego]
        z = self.agent_zone[ego]
        role = self.agent_role[ego]

        role_to_id = {
            self.ROLE_COLLECTOR: 0,
            self.ROLE_GATEKEEPER: 1,
            self.ROLE_RELAY: 2,
            self.ROLE_BLOCKER: 3,
        }

        obs = [
            own[0] / self.grid_size,
            own[1] / self.grid_size,
            z / max(1, self.n_zones - 1),
            role_to_id[role] / 3.0,
        ]

        gate = self.zone_gate[z]
        res = self.zone_resource[z]
        relay = self.zone_relay[z]

        obs.extend([
            (gate[0] - own[0]) / self.grid_size,
            (gate[1] - own[1]) / self.grid_size,
            (res[0] - own[0]) / self.grid_size,
            (res[1] - own[1]) / self.grid_size,
            (relay[0] - own[0]) / self.grid_size,
            (relay[1] - own[1]) / self.grid_size,
            float(self.gate_open[z]),
            float(self.resource_available[z]),
        ])

        for j in range(self.n_agents):
            if j == ego:
                continue
            pj = self.positions[j]
            obs.extend([
                (pj[0] - own[0]) / self.grid_size,
                (pj[1] - own[1]) / self.grid_size,
                self.agent_zone[j] / max(1, self.n_zones - 1),
                role_to_id[self.agent_role[j]] / 3.0,
                self.last_actions[j] / max(1, self.N_ACTIONS - 1),
            ])

        return np.array(obs, dtype=np.float32)

    def _get_obs_all(self):
        return [self._get_obs(i) for i in range(self.n_agents)]

    def rollout_from_current_state(self, forced=None, horizon=None):
        if horizon is None:
            horizon = self.causal_horizon

        snapshot = self.clone_state()
        gamma = 0.95
        total = np.zeros(self.n_agents, dtype=np.float32)

        done = False
        local_t = 0
        while (not done) and local_t < horizon:
            acts = [self.scripted_policy(i) for i in range(self.n_agents)]
            if forced is not None:
                aid, forced_action, forced_step = forced
                if local_t == forced_step:
                    acts[aid] = forced_action

            _, rewards, done, _ = self.step(acts)
            total += (gamma ** local_t) * np.array(rewards, dtype=np.float32)
            local_t += 1

        self.restore_state(snapshot)
        return total

    def compute_oracle_influence_from_current_state(self, ego_id, agent_j, intervention_action, horizon=None, n_trials=1):
        if horizon is None:
            horizon = self.causal_horizon

        snapshot = self.clone_state()
        deltas = []

        for _ in range(n_trials):
            self.restore_state(snapshot)
            base = self.rollout_from_current_state(forced=None, horizon=horizon)

            self.restore_state(snapshot)
            alt = self.rollout_from_current_state(
                forced=(agent_j, intervention_action, 0),
                horizon=horizon,
            )

            deltas.append(abs(float(alt[ego_id] - base[ego_id])))

        self.restore_state(snapshot)
        return float(np.mean(deltas))

    def sample_state_bank(self, n_states=24, burn_in=3):
        bank = []
        self.reset()
        for _ in range(n_states):
            for _ in range(burn_in):
                acts = [self.scripted_policy(i) for i in range(self.n_agents)]
                _, _, done, _ = self.step(acts)
                if done:
                    self.reset()
            bank.append(self.clone_state())
        return bank