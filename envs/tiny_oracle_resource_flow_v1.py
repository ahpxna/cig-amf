import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class TinyNode:
    node_id: int
    zone: int
    kind: str                  # source / buffer / lane / station / sink
    pos: Tuple[int, int]
    resource_type: int = -1
    product_type: int = -1
    capacity: int = 1
    queue_limit: int = 1
    hidden_priority: int = -1
    hidden_degraded: int = 0


class TinyOracleResourceFlowV1:
    """
    Tiny Oracle companion env cho AdaptiveResourceFlowArenaV3.

    Mục tiêu:
    - cùng ontology với env chính:
        source -> buffer -> lane -> station -> sink
    - nhỏ, clone-state sạch, can thiệp rẻ
    - dùng cho:
        sample_state_bank()
        compute_oracle_influence_from_current_state()
        estimate_oracle_core_from_current_state()

    Vai trò:
    - không phải benchmark chính để train dài
    - là oracle env để hiệu chuẩn local surrogate / structural score
    """

    # =========================================================
    # Actions
    # =========================================================
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    PICK = 5
    DROP_TO_BUFFER = 6
    PROCESS = 7
    DELIVER = 8
    TOGGLE_LANE = 9
    SIGNAL_A = 10
    SIGNAL_B = 11
    INSPECT = 12

    ACTION_NAMES = {
        0: "stay",
        1: "up",
        2: "down",
        3: "left",
        4: "right",
        5: "pick",
        6: "drop_to_buffer",
        7: "process",
        8: "deliver",
        9: "toggle_lane",
        10: "signal_a",
        11: "signal_b",
        12: "inspect",
    }

    # =========================================================
    # Tile codes
    # =========================================================
    TILE_EMPTY = 0
    TILE_WALL = 1
    TILE_SOURCE = 2
    TILE_BUFFER = 3
    TILE_STATION = 4
    TILE_SINK = 5
    TILE_LANE = 6
    TILE_OCCLUDER = 7

    ROLE_TO_CAP = {
        "hauler":     np.array([1.0, 0.4, 0.6, 0.8], dtype=np.float32),
        "processor":  np.array([0.5, 1.0, 0.5, 0.8], dtype=np.float32),
        "dispatcher": np.array([0.6, 0.5, 1.0, 1.0], dtype=np.float32),
        "spoiler":    np.array([0.7, 0.3, 0.9, 0.6], dtype=np.float32),
    }

    STYLES = [
        "cooperative",
        "greedy_localist",
        "opportunistic",
        "misleading_signaler",
    ]

    def __init__(
        self,
        n_agents: int = 6,
        grid_size: int = 9,
        n_zones: int = 2,
        max_steps: int = 20,
        obs_radius: int = 2,
        seed: int = 42,
    ):
        assert n_agents >= 4, "tiny env nên có ít nhất 4 agents"
        assert n_zones in [1, 2], "tiny env khuyên dùng 1 hoặc 2 zones"

        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_zones = n_zones
        self.max_steps = max_steps
        self.obs_radius = obs_radius

        self.rng = np.random.RandomState(seed)

        self.nodes: List[TinyNode] = []
        self.node_by_pos: Dict[Tuple[int, int], int] = {}
        self.walls = set()
        self.occluders = set()

        self.positions: List[List[int]] = []
        self.agent_zone: List[int] = []
        self.agent_role: List[str] = []
        self.agent_cap: List[np.ndarray] = []
        self.agent_style: List[str] = []

        self.inventory_type: List[int] = []
        self.inventory_qty: List[int] = []
        self.last_actions: List[int] = []
        self.last_signals: List[int] = []
        self.inspect_memory: List[Dict[str, float]] = []

        self.buffer_stock: Dict[int, int] = {}
        self.station_progress: Dict[int, float] = {}
        self.station_active_agent: Dict[int, int] = {}
        self.station_queue: Dict[int, List[int]] = {}
        self.station_recipe: Dict[int, Tuple[int, int]] = {}
        self.lane_open: Dict[int, int] = {}
        self.lane_controller: Dict[int, int] = {}
        self.sink_demand: Dict[int, int] = {}

        self.recent_interactions: List[List[Tuple[int, int, int]]] = []
        self.pair_trace_counts: Dict[Tuple[int, int], np.ndarray] = {}

        self.step_count = 0
        self.causal_horizon = 3

        self._build_layout()
        self.reset()

    # =========================================================
    # Layout
    # =========================================================
    def _build_layout(self):
        self.nodes = []
        self.node_by_pos = {}
        self.walls = set()
        self.occluders = set()

        for r in range(self.grid_size):
            self.walls.add((r, 0))
            self.walls.add((r, self.grid_size - 1))
        for c in range(self.grid_size):
            self.walls.add((0, c))
            self.walls.add((self.grid_size - 1, c))

        node_id = 0
        if self.n_zones == 1:
            centers = [self.grid_size // 2]
        else:
            centers = [2, self.grid_size - 3]

        for z, cc in enumerate(centers):
            source_pos = (1, cc)
            buffer_pos = (3, cc)
            lane_pos = (4, cc)
            station_pos = (5, cc)
            sink_pos = (7, cc)

            zone_nodes = [
                TinyNode(node_id=node_id, zone=z, kind="source", pos=source_pos, resource_type=z % 3, capacity=2),
                TinyNode(node_id=node_id + 1, zone=z, kind="buffer", pos=buffer_pos, capacity=3),
                TinyNode(node_id=node_id + 2, zone=z, kind="lane", pos=lane_pos, capacity=1),
                TinyNode(node_id=node_id + 3, zone=z, kind="station", pos=station_pos, queue_limit=2, capacity=1),
                TinyNode(node_id=node_id + 4, zone=z, kind="sink", pos=sink_pos, product_type=z % 3),
            ]
            for nd in zone_nodes:
                self.nodes.append(nd)
                self.node_by_pos[nd.pos] = nd.node_id
            node_id += 5

            # a bit of visibility asymmetry
            occ = (4, max(1, cc - 1))
            if occ not in self.node_by_pos:
                self.occluders.add(occ)

    # =========================================================
    # Reset
    # =========================================================
    def _assign_agents(self):
        self.positions = []
        self.agent_zone = []
        self.agent_role = []
        self.agent_cap = []
        self.agent_style = []

        roles = list(self.ROLE_TO_CAP.keys())
        styles = list(self.STYLES)

        free_cells = []
        for r in range(1, self.grid_size - 1):
            for c in range(1, self.grid_size - 1):
                pos = (r, c)
                if pos in self.walls or pos in self.occluders or pos in self.node_by_pos:
                    continue
                free_cells.append(pos)
        self.rng.shuffle(free_cells)

        for a in range(self.n_agents):
            pos = list(free_cells[a])
            zone = 0 if self.n_zones == 1 else (0 if pos[1] < self.grid_size // 2 else 1)
            role = roles[a % len(roles)]
            style = styles[a % len(styles)]

            self.positions.append(pos)
            self.agent_zone.append(zone)
            self.agent_role.append(role)
            self.agent_cap.append(self.ROLE_TO_CAP[role].copy())
            self.agent_style.append(style)

    def _init_dynamic_state(self):
        self.inventory_type = [-1 for _ in range(self.n_agents)]
        self.inventory_qty = [0 for _ in range(self.n_agents)]
        self.last_actions = [self.STAY for _ in range(self.n_agents)]
        self.last_signals = [0 for _ in range(self.n_agents)]
        self.inspect_memory = [dict() for _ in range(self.n_agents)]

        self.buffer_stock = {}
        self.station_progress = {}
        self.station_active_agent = {}
        self.station_queue = {}
        self.station_recipe = {}
        self.lane_open = {}
        self.lane_controller = {}
        self.sink_demand = {}

        self.recent_interactions = [[] for _ in range(self.n_agents)]
        self.pair_trace_counts = {
            (i, j): np.zeros(6, dtype=np.float32)
            for i in range(self.n_agents)
            for j in range(self.n_agents)
            if i != j
        }

        for nd in self.nodes:
            if nd.kind == "buffer":
                self.buffer_stock[nd.node_id] = 0
            elif nd.kind == "station":
                self.station_progress[nd.node_id] = 0.0
                self.station_active_agent[nd.node_id] = -1
                self.station_queue[nd.node_id] = []
                self.station_recipe[nd.node_id] = (nd.zone % 3, nd.zone % 3)
            elif nd.kind == "lane":
                self.lane_open[nd.node_id] = 1
                self.lane_controller[nd.node_id] = -1
            elif nd.kind == "sink":
                self.sink_demand[nd.node_id] = nd.zone % 3

        self._sample_hidden_rules()

    def _sample_hidden_rules(self):
        for nd in self.nodes:
            if nd.kind == "lane":
                nd.hidden_priority = int(self.rng.randint(0, self.n_agents))
                self.lane_controller[nd.node_id] = nd.hidden_priority
                self.lane_open[nd.node_id] = 1
            elif nd.kind == "station":
                nd.hidden_degraded = int(self.rng.rand() < 0.30)
                raw_t = int(self.rng.randint(0, 3))
                prod_t = int(self.rng.randint(0, 3))
                self.station_recipe[nd.node_id] = (raw_t, prod_t)
            elif nd.kind == "sink":
                self.sink_demand[nd.node_id] = int(self.rng.randint(0, 3))

    def reset(self):
        self.step_count = 0
        self._assign_agents()
        self._init_dynamic_state()
        return self._get_obs_all()

    # =========================================================
    # Step
    # =========================================================
    def step(self, actions: List[int]):
        assert len(actions) == self.n_agents

        rewards = np.zeros(self.n_agents, dtype=np.float32)

        proposed = [self._propose_move(a, int(actions[a])) for a in range(self.n_agents)]
        for a in range(self.n_agents):
            self.positions[a] = proposed[a]

        for a in range(self.n_agents):
            act = int(actions[a])
            self.last_actions[a] = act
            self.last_signals[a] = 0

            if act == self.SIGNAL_A:
                self.last_signals[a] = 1
                rewards[a] -= 0.002
            elif act == self.SIGNAL_B:
                self.last_signals[a] = 2
                rewards[a] -= 0.002
            elif act == self.PICK:
                rewards[a] += self._handle_pick(a)
            elif act == self.DROP_TO_BUFFER:
                rewards[a] += self._handle_drop_to_buffer(a)
            elif act == self.PROCESS:
                rewards[a] += self._handle_process(a)
            elif act == self.DELIVER:
                rewards[a] += self._handle_deliver(a)
            elif act == self.TOGGLE_LANE:
                rewards[a] += self._handle_toggle_lane(a)
            elif act == self.INSPECT:
                rewards[a] += self._handle_inspect(a)
            elif act == self.STAY:
                rewards[a] -= 0.005

        rewards += self._advance_station_processing()
        rewards += self._apply_congestion_penalties()
        rewards += self._throughput_shaping()

        self._update_recent_interactions(actions)
        self._update_pair_traces(actions)

        self.step_count += 1
        done = self.step_count >= self.max_steps
        info = {}
        return self._get_obs_all(), rewards.tolist(), done, info

    # =========================================================
    # Core mechanics
    # =========================================================
    def _propose_move(self, agent_id: int, action: int):
        r, c = self.positions[agent_id]
        nr, nc = r, c
        if action == self.UP:
            nr -= 1
        elif action == self.DOWN:
            nr += 1
        elif action == self.LEFT:
            nc -= 1
        elif action == self.RIGHT:
            nc += 1

        if (nr, nc) in self.walls:
            return [r, c]
        nr = max(1, min(self.grid_size - 2, nr))
        nc = max(1, min(self.grid_size - 2, nc))
        return [nr, nc]

    def _adjacent_nodes(self, agent_id: int, kind: Optional[str] = None):
        r, c = self.positions[agent_id]
        out = []
        for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            pos = (r + dr, c + dc)
            if pos in self.node_by_pos:
                nd = self.nodes[self.node_by_pos[pos]]
                if kind is None or nd.kind == kind:
                    out.append(nd)
        return out

    def _handle_pick(self, agent_id: int) -> float:
        if self.inventory_qty[agent_id] > 0:
            return -0.02
        srcs = self._adjacent_nodes(agent_id, "source")
        if not srcs:
            return -0.02

        src = srcs[0]
        self.inventory_type[agent_id] = src.resource_type
        self.inventory_qty[agent_id] = 1
        return 0.03

    def _handle_drop_to_buffer(self, agent_id: int) -> float:
        if self.inventory_qty[agent_id] <= 0:
            return -0.02
        bufs = self._adjacent_nodes(agent_id, "buffer")
        if not bufs:
            return -0.02

        buf = bufs[0]
        st = self._get_zone_station(buf.zone)
        raw_t, _ = self.station_recipe[st.node_id]

        self.buffer_stock[buf.node_id] = min(buf.capacity, self.buffer_stock[buf.node_id] + 1)
        rew = 0.08 if self.inventory_type[agent_id] == raw_t else -0.08

        self.inventory_type[agent_id] = -1
        self.inventory_qty[agent_id] = 0
        return rew

    def _handle_process(self, agent_id: int) -> float:
        sts = self._adjacent_nodes(agent_id, "station")
        if not sts:
            return -0.02
        st = sts[0]
        buf = self._get_zone_buffer(st.zone)
        lane = self._get_zone_lane(st.zone)

        privileged = (self.lane_controller[lane.node_id] == agent_id)
        strong_lane = self.agent_cap[agent_id][2] > 0.85
        if not self.lane_open[lane.node_id] and not privileged and not strong_lane:
            return -0.05

        if st.hidden_degraded and self.agent_cap[agent_id][1] < 0.8:
            return -0.04

        if self.buffer_stock[buf.node_id] <= 0:
            return -0.03

        if agent_id == self.station_active_agent[st.node_id]:
            return 0.0
        if agent_id in self.station_queue[st.node_id]:
            return 0.0

        if self.station_active_agent[st.node_id] == -1:
            self.buffer_stock[buf.node_id] -= 1
            self.station_active_agent[st.node_id] = agent_id
            self.station_progress[st.node_id] = 0.0
            return 0.05

        if len(self.station_queue[st.node_id]) >= st.queue_limit:
            return -0.03

        self.station_queue[st.node_id].append(agent_id)
        return 0.01

    def _advance_station_processing(self) -> np.ndarray:
        rew = np.zeros(self.n_agents, dtype=np.float32)

        for nd in self.nodes:
            if nd.kind != "station":
                continue

            sid = nd.node_id
            active = self.station_active_agent[sid]
            if active == -1:
                continue

            speed = 1.0 + 0.5 * float(self.agent_cap[active][1])
            if nd.hidden_degraded:
                speed *= 0.6

            self.station_progress[sid] += speed

            if self.station_progress[sid] >= 2.0:
                _, prod_t = self.station_recipe[sid]
                if self.inventory_qty[active] == 0:
                    self.inventory_type[active] = prod_t
                    self.inventory_qty[active] = 1
                    rew[active] += 0.18 if not nd.hidden_degraded else 0.10
                else:
                    rew[active] -= 0.02

                if len(self.station_queue[sid]) > 0:
                    nxt = self.station_queue[sid].pop(0)
                    buf = self._get_zone_buffer(nd.zone)
                    if self.buffer_stock[buf.node_id] > 0:
                        self.buffer_stock[buf.node_id] -= 1
                        self.station_active_agent[sid] = nxt
                        self.station_progress[sid] = 0.0
                    else:
                        self.station_active_agent[sid] = -1
                        self.station_progress[sid] = 0.0
                else:
                    self.station_active_agent[sid] = -1
                    self.station_progress[sid] = 0.0

        return rew

    def _handle_deliver(self, agent_id: int) -> float:
        if self.inventory_qty[agent_id] <= 0:
            return -0.02
        sinks = self._adjacent_nodes(agent_id, "sink")
        if not sinks:
            return -0.02

        sk = sinks[0]
        demand = self.sink_demand[sk.node_id]
        rew = 1.2 if self.inventory_type[agent_id] == demand else -0.25

        self.inventory_type[agent_id] = -1
        self.inventory_qty[agent_id] = 0
        return rew

    def _handle_toggle_lane(self, agent_id: int) -> float:
        lanes = self._adjacent_nodes(agent_id, "lane")
        if not lanes:
            return -0.01
        lane = lanes[0]

        privileged = self.lane_controller[lane.node_id] == agent_id
        strong_lane = self.agent_cap[agent_id][2] > 0.85
        if privileged:
            self.lane_open[lane.node_id] = 1 - self.lane_open[lane.node_id]
            return 0.05
        if strong_lane:
            self.lane_open[lane.node_id] = 1 - self.lane_open[lane.node_id]
            return 0.02
        return -0.03

    def _handle_inspect(self, agent_id: int) -> float:
        nearby = self._adjacent_nodes(agent_id, None)
        if not nearby:
            return -0.01

        rew = 0.0
        for nd in nearby:
            if nd.kind == "lane":
                self.inspect_memory[agent_id][f"lane_controller_zone_{nd.zone}"] = float(self.lane_controller[nd.node_id]) / max(1, self.n_agents - 1)
                rew += 0.01
            elif nd.kind == "station":
                self.inspect_memory[agent_id][f"station_degraded_zone_{nd.zone}"] = float(nd.hidden_degraded)
                raw_t, prod_t = self.station_recipe[nd.node_id]
                self.inspect_memory[agent_id][f"station_raw_zone_{nd.zone}"] = float(raw_t) / 2.0
                self.inspect_memory[agent_id][f"station_prod_zone_{nd.zone}"] = float(prod_t) / 2.0
                rew += 0.01
            elif nd.kind == "sink":
                self.inspect_memory[agent_id][f"sink_demand_zone_{nd.zone}"] = float(self.sink_demand[nd.node_id]) / 2.0
                rew += 0.01
        return rew

    def _apply_congestion_penalties(self) -> np.ndarray:
        rew = np.zeros(self.n_agents, dtype=np.float32)
        pos_counts: Dict[Tuple[int, int], List[int]] = {}
        for a, p in enumerate(self.positions):
            pos_counts.setdefault(tuple(p), []).append(a)

        for pos, ids in pos_counts.items():
            if len(ids) <= 1:
                continue

            if pos in self.node_by_pos and self.nodes[self.node_by_pos[pos]].kind == "lane":
                lane = self.nodes[self.node_by_pos[pos]]
                controller = self.lane_controller[lane.node_id]
                for a in ids:
                    rew[a] -= 0.01 if a == controller else 0.08
            else:
                for a in ids:
                    rew[a] -= 0.03
        return rew

    def _throughput_shaping(self) -> np.ndarray:
        rew = np.zeros(self.n_agents, dtype=np.float32)
        for z in range(self.n_zones):
            buf = self._get_zone_buffer(z)
            lane = self._get_zone_lane(z)
            if self.buffer_stock[buf.node_id] > 0 and self.lane_open[lane.node_id]:
                for a in range(self.n_agents):
                    if self.agent_zone[a] == z:
                        rew[a] += 0.002
        return rew

    # =========================================================
    # Histories / traces
    # =========================================================
    def _update_recent_interactions(self, actions: List[int]):
        for ego in range(self.n_agents):
            er, ec = self.positions[ego]
            local = []
            for j in range(self.n_agents):
                if j == ego:
                    continue
                jr, jc = self.positions[j]
                if abs(er - jr) + abs(ec - jc) <= 2:
                    local.append((j, int(actions[j]), int(self.last_signals[j])))
            self.recent_interactions[ego].extend(local)
            self.recent_interactions[ego] = self.recent_interactions[ego][-12:]

    def _update_pair_traces(self, actions: List[int]):
        lane_positions = {nd.pos for nd in self.nodes if nd.kind == "lane"}
        buffer_positions = {nd.pos for nd in self.nodes if nd.kind == "buffer"}
        sink_positions = {nd.pos for nd in self.nodes if nd.kind == "sink"}

        for i in range(self.n_agents):
            pi = tuple(self.positions[i])
            for j in range(self.n_agents):
                if i == j:
                    continue
                pj = tuple(self.positions[j])
                tr = self.pair_trace_counts[(i, j)]

                dist = abs(pi[0] - pj[0]) + abs(pi[1] - pj[1])
                if dist <= 2:
                    tr[0] += 1.0
                if pi in lane_positions and pj in lane_positions:
                    tr[1] += 1.0
                if pi in buffer_positions and pj in buffer_positions:
                    tr[2] += 1.0
                if self.last_signals[j] > 0:
                    tr[3] += 1.0

                if pi == pj and pi in lane_positions:
                    controller = self._lane_controller_at_pos(pi)
                    if controller == j or self.agent_cap[j][2] > self.agent_cap[i][2]:
                        tr[4] += 1.0

                if pi in sink_positions and pj in sink_positions and self.agent_zone[i] != self.agent_zone[j]:
                    tr[5] += 1.0

    # =========================================================
    # Scripted default policies
    # =========================================================
    def scripted_policy(self, agent_id: int) -> int:
        style = self.agent_style[agent_id]
        zone = self.agent_zone[agent_id]
        pos = tuple(self.positions[agent_id])

        src = self._get_zone_source(zone)
        buf = self._get_zone_buffer(zone)
        st = self._get_zone_station(zone)
        sk = self._get_zone_sink(zone)
        lane = self._get_zone_lane(zone)

        if style == "cooperative":
            if self.inventory_qty[agent_id] == 0 and self._manhattan(pos, src.pos) <= 1:
                return self.PICK
            if self.inventory_qty[agent_id] > 0 and self._manhattan(pos, buf.pos) <= 1:
                return self.DROP_TO_BUFFER
            if self.inventory_qty[agent_id] == 0 and self._manhattan(pos, st.pos) <= 1 and self.buffer_stock[buf.node_id] > 0:
                return self.PROCESS
            if self.inventory_qty[agent_id] > 0 and self._manhattan(pos, sk.pos) <= 1:
                return self.DELIVER
            return self._greedy_move_toward(agent_id, src.pos if self.inventory_qty[agent_id] == 0 else buf.pos)

        if style == "greedy_localist":
            if self.inventory_qty[agent_id] > 0 and self._manhattan(pos, sk.pos) <= 1:
                return self.DELIVER
            return self._greedy_move_toward(agent_id, sk.pos)

        if style == "opportunistic":
            if self._manhattan(pos, lane.pos) <= 1:
                return self.TOGGLE_LANE
            return self._greedy_move_toward(agent_id, lane.pos)

        if style == "misleading_signaler":
            if self._manhattan(pos, lane.pos) <= 2:
                return self.SIGNAL_B
            return self._greedy_move_toward(agent_id, lane.pos)

        return self.STAY

    def default_joint_action(self) -> List[int]:
        return [self.scripted_policy(a) for a in range(self.n_agents)]

    # =========================================================
    # Oracle utilities
    # =========================================================
    def rollout_from_current_state(
        self,
        forced: Optional[Dict[int, int]] = None,
        horizon: int = 3,
    ) -> np.ndarray:
        saved = self.clone_state()
        out = np.zeros(self.n_agents, dtype=np.float32)

        gamma = 0.95
        forced = forced or {}

        for h in range(horizon):
            acts = self.default_joint_action()
            for aid, act in forced.items():
                acts[int(aid)] = int(act)

            _, rewards, done, _ = self.step(acts)
            out += (gamma ** h) * np.array(rewards, dtype=np.float32)
            if done:
                break

        self.restore_state(saved)
        return out

    def compute_oracle_influence_from_current_state(
        self,
        ego_id: int,
        agent_j: int,
        intervention_action: int,
        horizon: int = 3,
        n_trials: int = 3,
    ) -> float:
        """
        Local oracle effect:
            E[ R_i(do(a_j=a')) - R_i(baseline) ]
        """
        saved = self.clone_state()
        vals = []

        for _ in range(n_trials):
            self.restore_state(saved)
            base = self.rollout_from_current_state(forced=None, horizon=horizon)

            self.restore_state(saved)
            inter = self.rollout_from_current_state(
                forced={agent_j: intervention_action},
                horizon=horizon,
            )

            vals.append(float(inter[ego_id] - base[ego_id]))

        self.restore_state(saved)
        return float(np.mean(vals))

    def estimate_oracle_core_from_current_state(
        self,
        ego_id: int,
        horizon: int = 3,
        n_trials: int = 2,
        top_k: int = 2,
    ) -> Tuple[Dict[int, float], List[int]]:
        scores = {}
        candidate_actions = [self.STAY, self.SIGNAL_A, self.SIGNAL_B, self.TOGGLE_LANE]

        for j in range(self.n_agents):
            if j == ego_id:
                continue
            vals = []
            for a in candidate_actions:
                vals.append(abs(self.compute_oracle_influence_from_current_state(
                    ego_id=ego_id,
                    agent_j=j,
                    intervention_action=a,
                    horizon=horizon,
                    n_trials=n_trials,
                )))
            scores[j] = float(np.mean(vals))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = [j for j, _ in ranked[:top_k]]
        return scores, top

    def sample_state_bank(
        self,
        n_states: int = 16,
        burn_in: int = 3,
    ) -> List[dict]:
        """
        Tạo bank các trạng thái "sống" từ scripted traffic.
        Mỗi state lấy sau vài bước burn-in để có queue / inventory / signal thật hơn.
        """
        bank = []
        attempts = 0
        while len(bank) < n_states and attempts < n_states * 10:
            attempts += 1
            self.reset()

            for _ in range(burn_in):
                acts = self.default_joint_action()
                _, _, done, _ = self.step(acts)
                if done:
                    break

            bank.append(self.clone_state())

        return bank

    def get_supported_egos(self) -> List[int]:
        """
        Tiny env không cần dùng hết mọi ego; ta chọn vài ego đại diện.
        """
        return list(range(min(3, self.n_agents)))

    # =========================================================
    # Observation
    # =========================================================
    def _get_obs_all(self):
        return [self._get_obs(a) for a in range(self.n_agents)]

    def get_obs_of_ego(self, obs_all, ego_id: int):
        return obs_all[ego_id]

    def _get_obs(self, agent_id: int):
        r0, c0 = self.positions[agent_id]
        sr = self.obs_radius

        patch = []
        for dr in range(-sr, sr + 1):
            for dc in range(-sr, sr + 1):
                rr, cc = r0 + dr, c0 + dc
                patch.append(self._tile_code(rr, cc) / 7.0)

        visible = [j for j in self._visible_agents_for(agent_id) if j != agent_id]
        visible = sorted(
            visible,
            key=lambda j: abs(self.positions[j][0] - r0) + abs(self.positions[j][1] - c0)
        )[:4]

        nearby_feats = []
        for j in visible:
            rr = (self.positions[j][0] - r0) / max(1, self.grid_size)
            cc = (self.positions[j][1] - c0) / max(1, self.grid_size)
            nearby_feats.extend([
                rr,
                cc,
                float(self.agent_zone[j]) / max(1, self.n_zones - 1),
                float(self.inventory_type[j] + 1) / 3.0,
                float(self.last_signals[j]) / 2.0,
                float(self.last_actions[j]) / max(1, self.get_action_dim() - 1),
            ])
        while len(nearby_feats) < 4 * 6:
            nearby_feats.extend([0.0] * 6)

        public = []
        for z in range(self.n_zones):
            buf = self._get_zone_buffer(z)
            st = self._get_zone_station(z)
            sk = self._get_zone_sink(z)
            lane = self._get_zone_lane(z)
            public.extend([
                float(self.buffer_stock[buf.node_id]) / max(1, buf.capacity),
                float(len(self.station_queue[st.node_id])) / max(1, st.queue_limit),
                float(self.station_active_agent[st.node_id] != -1),
                float(self.lane_open[lane.node_id]),
                float(self.sink_demand[sk.node_id]) / 2.0,
            ])

        private = []
        private.extend(self.agent_cap[agent_id].tolist())
        private.extend([
            float(self.inventory_type[agent_id] + 1) / 3.0,
            float(self.inventory_qty[agent_id]),
            float(self.agent_zone[agent_id]) / max(1, self.n_zones - 1),
        ])

        mem = []
        keys = []
        for z in range(self.n_zones):
            keys.extend([
                f"lane_controller_zone_{z}",
                f"station_degraded_zone_{z}",
                f"station_raw_zone_{z}",
                f"station_prod_zone_{z}",
                f"sink_demand_zone_{z}",
            ])
        for k in keys:
            mem.append(float(self.inspect_memory[agent_id].get(k, -1.0)))

        pair_feats = []
        candidates = sorted(
            [j for j in range(self.n_agents) if j != agent_id],
            key=lambda j: abs(self.positions[j][0] - r0) + abs(self.positions[j][1] - c0),
        )[:4]

        denom = max(1, self.step_count + 1)
        for j in candidates:
            pair_feats.extend((self.pair_trace_counts[(agent_id, j)] / denom).tolist())
        while len(pair_feats) < 4 * 6:
            pair_feats.extend([0.0] * 6)

        return np.array(
            patch + nearby_feats + public + private + mem + pair_feats,
            dtype=np.float32,
        )

    def _visible_agents_for(self, agent_id: int):
        r0, c0 = self.positions[agent_id]
        out = []
        for j in range(self.n_agents):
            r1, c1 = self.positions[j]
            if max(abs(r1 - r0), abs(c1 - c0)) > self.obs_radius:
                continue

            hidden = False
            if (r1, c1) in self.occluders or (r0, c0) in self.occluders:
                hidden = True

            if hidden and self.agent_cap[agent_id][3] < 0.95 and agent_id != j:
                continue
            out.append(j)
        return out

    def _tile_code(self, r: int, c: int):
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return self.TILE_WALL
        pos = (r, c)
        if pos in self.walls:
            return self.TILE_WALL
        if pos in self.occluders:
            return self.TILE_OCCLUDER
        if pos in self.node_by_pos:
            nd = self.nodes[self.node_by_pos[pos]]
            return {
                "source": self.TILE_SOURCE,
                "buffer": self.TILE_BUFFER,
                "station": self.TILE_STATION,
                "sink": self.TILE_SINK,
                "lane": self.TILE_LANE,
            }[nd.kind]
        return self.TILE_EMPTY

    # =========================================================
    # Helpers
    # =========================================================
    def _get_zone_source(self, zone: int) -> TinyNode:
        return next(nd for nd in self.nodes if nd.zone == zone and nd.kind == "source")

    def _get_zone_buffer(self, zone: int) -> TinyNode:
        return next(nd for nd in self.nodes if nd.zone == zone and nd.kind == "buffer")

    def _get_zone_station(self, zone: int) -> TinyNode:
        return next(nd for nd in self.nodes if nd.zone == zone and nd.kind == "station")

    def _get_zone_sink(self, zone: int) -> TinyNode:
        return next(nd for nd in self.nodes if nd.zone == zone and nd.kind == "sink")

    def _get_zone_lane(self, zone: int) -> TinyNode:
        return next(nd for nd in self.nodes if nd.zone == zone and nd.kind == "lane")

    def _lane_controller_at_pos(self, pos: Tuple[int, int]) -> int:
        if pos not in self.node_by_pos:
            return -1
        nd = self.nodes[self.node_by_pos[pos]]
        if nd.kind != "lane":
            return -1
        return self.lane_controller[nd.node_id]

    def _manhattan(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _greedy_move_toward(self, agent_id: int, target: Tuple[int, int]) -> int:
        r, c = self.positions[agent_id]
        tr, tc = target
        if abs(tr - r) > abs(tc - c):
            return self.DOWN if tr > r else self.UP
        if tc != c:
            return self.RIGHT if tc > c else self.LEFT
        if tr != r:
            return self.DOWN if tr > r else self.UP
        return self.STAY

    # =========================================================
    # API dimensions
    # =========================================================
    def get_obs_dim(self):
        patch_dim = (2 * self.obs_radius + 1) ** 2
        nearby_dim = 4 * 6
        public_dim = self.n_zones * 5
        private_dim = 7
        mem_dim = self.n_zones * 5
        pair_dim = 4 * 6
        return patch_dim + nearby_dim + public_dim + private_dim + mem_dim + pair_dim

    def get_action_dim(self):
        return 13

    # =========================================================
    # Clone / restore
    # =========================================================
    def clone_state(self):
        return copy.deepcopy({
            "positions": self.positions,
            "agent_zone": self.agent_zone,
            "agent_role": self.agent_role,
            "agent_cap": self.agent_cap,
            "agent_style": self.agent_style,
            "inventory_type": self.inventory_type,
            "inventory_qty": self.inventory_qty,
            "last_actions": self.last_actions,
            "last_signals": self.last_signals,
            "inspect_memory": self.inspect_memory,
            "buffer_stock": self.buffer_stock,
            "station_progress": self.station_progress,
            "station_active_agent": self.station_active_agent,
            "station_queue": self.station_queue,
            "station_recipe": self.station_recipe,
            "lane_open": self.lane_open,
            "lane_controller": self.lane_controller,
            "sink_demand": self.sink_demand,
            "recent_interactions": self.recent_interactions,
            "pair_trace_counts": self.pair_trace_counts,
            "nodes": self.nodes,
            "step_count": self.step_count,
            "rng_state": self.rng.get_state(),
        })

    def restore_state(self, state):
        self.positions = copy.deepcopy(state["positions"])
        self.agent_zone = copy.deepcopy(state["agent_zone"])
        self.agent_role = copy.deepcopy(state["agent_role"])
        self.agent_cap = copy.deepcopy(state["agent_cap"])
        self.agent_style = copy.deepcopy(state["agent_style"])
        self.inventory_type = copy.deepcopy(state["inventory_type"])
        self.inventory_qty = copy.deepcopy(state["inventory_qty"])
        self.last_actions = copy.deepcopy(state["last_actions"])
        self.last_signals = copy.deepcopy(state["last_signals"])
        self.inspect_memory = copy.deepcopy(state["inspect_memory"])
        self.buffer_stock = copy.deepcopy(state["buffer_stock"])
        self.station_progress = copy.deepcopy(state["station_progress"])
        self.station_active_agent = copy.deepcopy(state["station_active_agent"])
        self.station_queue = copy.deepcopy(state["station_queue"])
        self.station_recipe = copy.deepcopy(state["station_recipe"])
        self.lane_open = copy.deepcopy(state["lane_open"])
        self.lane_controller = copy.deepcopy(state["lane_controller"])
        self.sink_demand = copy.deepcopy(state["sink_demand"])
        self.recent_interactions = copy.deepcopy(state["recent_interactions"])
        self.pair_trace_counts = copy.deepcopy(state["pair_trace_counts"])
        self.nodes = copy.deepcopy(state["nodes"])
        self.step_count = state["step_count"]
        self.rng.set_state(state["rng_state"])