import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class Node:
    node_id: int
    zone: int
    kind: str
    pos: Tuple[int, int]
    resource_type: int = -1
    product_type: int = -1
    capacity: int = 1
    queue_limit: int = 1
    hidden_priority: int = -1
    hidden_degraded: int = 0


class AdaptiveResourceFlowArena:
    """
    Main adaptive resource-flow benchmark environment for CIG-AMF.

    Environment ontology:
        source -> buffer -> lane -> station -> sink

    Core design:
    - local, bounded, structured dependencies
    - hidden institutional rules
    - lane bottlenecks
    - station degradation
    - role/capability heterogeneity
    - partial observability
    - signalling
    - queueing/congestion
    - diagnostic structural core, not true causal graph
    - clone/restore support for intervention-style evaluation

    Important:
    diagnostic_core_by_ego is a diagnostic benchmark target only.
    It is not claimed to be a true causal graph.
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

    # =========================================================
    # Roles / capabilities
    # [carry, process, lane_control, sensing]
    # =========================================================

    ROLE_TO_CAP = {
        "hauler": np.array([1.0, 0.4, 0.6, 0.8], dtype=np.float32),
        "processor": np.array([0.5, 1.0, 0.5, 0.8], dtype=np.float32),
        "dispatcher": np.array([0.6, 0.5, 1.0, 1.0], dtype=np.float32),
        "sweeper": np.array([0.8, 0.6, 0.7, 0.7], dtype=np.float32),
        "spoiler": np.array([0.7, 0.3, 0.9, 0.6], dtype=np.float32),
    }

    STYLES = [
        "cooperative",
        "greedy_localist",
        "opportunistic",
        "conservative",
        "misleading_signaler",
        "retaliatory",
    ]

    # =========================================================
    # Init
    # =========================================================

    def __init__(
        self,
        n_agents: int = 16,
        grid_size: int = 17,
        n_zones: int = 4,
        max_steps: int = 60,
        obs_radius: int = 3,
        phase_length: int = 20,
        mode: str = "behavioral_drift",
        task_mode: Optional[str] = None,
        seed: int = 42,
        diagnostic_core_k: Optional[int] = None,
        max_core_size: Optional[int] = None,
        resample_agent_layout_each_reset: bool = False,
        resample_hidden_rules_each_reset: bool = False,
    ):
        if task_mode is not None:
            mode = task_mode

        assert int(n_agents) >= 2 * int(n_zones), "n_agents quá nhỏ so với n_zones"

        self.n_agents = int(n_agents)
        self.grid_size = int(grid_size)
        self.n_zones = int(n_zones)
        self.max_steps = int(max_steps)
        self.obs_radius = int(obs_radius)
        self.phase_length = int(phase_length)
        self.mode = str(mode)

        self.rng = np.random.RandomState(int(seed))

        self.zone_width = max(3, (self.grid_size - 2) // self.n_zones)
        self.max_visible_agents = 6
        self.max_pair_trace_neighbors = 6

        self.episode_count = 0
        self.current_phase = 0
        self.step_count = 0

        default_k = min(4, max(1, self.n_agents - 1))
        self.diagnostic_core_k = int(default_k if diagnostic_core_k is None else diagnostic_core_k)
        self.max_core_size = int(self.diagnostic_core_k if max_core_size is None else max_core_size)
        self.resample_agent_layout_each_reset = bool(resample_agent_layout_each_reset)
        self.resample_hidden_rules_each_reset = bool(resample_hidden_rules_each_reset)

        # Static world.
        self.nodes: List[Node] = []
        self.node_by_pos: Dict[Tuple[int, int], int] = {}
        self.walls = set()
        self.occluders = set()

        # Agents.
        self.positions: List[List[int]] = []
        self.agent_zone: List[int] = []
        self.agent_role: List[str] = []
        self.agent_cap: List[np.ndarray] = []
        self.agent_style: List[str] = []

        # Dynamic world state.
        self.inventory_type: List[int] = []
        self.inventory_qty: List[int] = []
        self.last_actions: List[int] = []
        self.last_signals: List[int] = []
        self.inspect_memory: List[Dict[str, float]] = []

        self.buffer_stock: Dict[int, int] = {}
        self.station_progress: Dict[int, float] = {}
        self.station_active_agent: Dict[int, int] = {}
        self.station_recipe: Dict[int, Tuple[int, int]] = {}
        self.station_queue: Dict[int, List[int]] = {}
        self.lane_open: Dict[int, int] = {}
        self.lane_controller: Dict[int, int] = {}
        self.sink_demand: Dict[int, int] = {}

        # Diagnostics / evaluation.
        self.diagnostic_core_by_ego: Dict[int, set] = {}
        self.recent_interactions: List[List[Tuple[int, int, int]]] = []
        self.pair_trace_counts: Dict[Tuple[int, int], np.ndarray] = {}

        self._build_static_layout()
        self.reset()

    # =========================================================
    # Seed / mode helpers
    # =========================================================

    def seed(self, seed: int):
        self.rng = np.random.RandomState(int(seed))

    def set_seed(self, seed: int):
        self.seed(seed)

    def set_mode(self, mode: str):
        self.mode = str(mode)

    def get_structure_regime_id(self):
        """Expose the environment-owned structural phase to causal replay."""
        return int(self.current_phase) if self.mode == "structural_shift" else 0

    # =========================================================
    # Static layout
    # =========================================================

    def _build_static_layout(self):
        self.nodes = []
        self.node_by_pos = {}
        self.walls = set()
        self.occluders = set()

        node_id = 0

        for r in range(self.grid_size):
            self.walls.add((r, 0))
            self.walls.add((r, self.grid_size - 1))

        for c in range(self.grid_size):
            self.walls.add((0, c))
            self.walls.add((self.grid_size - 1, c))

        for z in range(self.n_zones):
            c0 = 1 + z * self.zone_width
            c1 = min(self.grid_size - 2, c0 + self.zone_width - 1)
            cc = (c0 + c1) // 2

            source_pos = (2, cc)
            buffer_pos = (self.grid_size // 2 - 2, cc)
            lane_pos = (self.grid_size // 2 - 1, cc)
            station_pos = (self.grid_size // 2 + 1, cc)
            sink_pos = (self.grid_size - 3, cc)

            zone_nodes = [
                Node(
                    node_id=node_id,
                    zone=z,
                    kind="source",
                    pos=source_pos,
                    resource_type=z % 3,
                    capacity=3,
                ),
                Node(
                    node_id=node_id + 1,
                    zone=z,
                    kind="buffer",
                    pos=buffer_pos,
                    capacity=4,
                ),
                Node(
                    node_id=node_id + 2,
                    zone=z,
                    kind="lane",
                    pos=lane_pos,
                    capacity=1,
                ),
                Node(
                    node_id=node_id + 3,
                    zone=z,
                    kind="station",
                    pos=station_pos,
                    queue_limit=2,
                    capacity=1,
                ),
                Node(
                    node_id=node_id + 4,
                    zone=z,
                    kind="sink",
                    pos=sink_pos,
                    product_type=z % 3,
                ),
            ]

            for nd in zone_nodes:
                self.nodes.append(nd)
                self.node_by_pos[nd.pos] = nd.node_id

            node_id += len(zone_nodes)

            left_occ = (self.grid_size // 2, max(1, cc - 1))
            right_occ = (self.grid_size // 2, min(self.grid_size - 2, cc + 1))

            if left_occ not in self.node_by_pos:
                self.occluders.add(left_occ)

            if right_occ not in self.node_by_pos:
                self.occluders.add(right_occ)

        for r in range(4, self.grid_size - 3, 4):
            for c in range(2, self.grid_size - 2):
                pos = (r, c)

                if pos in self.node_by_pos or pos in self.occluders:
                    continue

                if c % 5 == 0:
                    self.walls.add(pos)

    # =========================================================
    # Reset / regimes
    # =========================================================

    def _assign_agents(self, force: bool = False):
        if (
            self.positions
            and not bool(force)
            and not bool(self.resample_agent_layout_each_reset)
        ):
            return

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

        if len(free_cells) < self.n_agents:
            raise RuntimeError(
                f"Not enough free cells for {self.n_agents} agents; only {len(free_cells)} free cells."
            )

        for a in range(self.n_agents):
            pos = list(free_cells[a])
            zone = min(
                self.n_zones - 1,
                max(0, (pos[1] - 1) // max(1, self.zone_width)),
            )
            role = roles[a % len(roles)]
            style = styles[a % len(styles)]

            self.positions.append(pos)
            self.agent_zone.append(int(zone))
            self.agent_role.append(role)
            self.agent_cap.append(self.ROLE_TO_CAP[role].copy())
            self.agent_style.append(style)

    def _init_dynamic_state(self, sample_hidden_rules: bool = True):
        prev_station_recipe = copy.deepcopy(getattr(self, "station_recipe", {}))
        prev_lane_controller = copy.deepcopy(getattr(self, "lane_controller", {}))
        prev_sink_demand = copy.deepcopy(getattr(self, "sink_demand", {}))

        self.inventory_type = [-1 for _ in range(self.n_agents)]
        self.inventory_qty = [0 for _ in range(self.n_agents)]
        self.last_actions = [self.STAY for _ in range(self.n_agents)]
        self.last_signals = [0 for _ in range(self.n_agents)]
        self.inspect_memory = [dict() for _ in range(self.n_agents)]

        self.recent_interactions = [[] for _ in range(self.n_agents)]
        self.pair_trace_counts = {
            (i, j): np.zeros(6, dtype=np.float32)
            for i in range(self.n_agents)
            for j in range(self.n_agents)
            if i != j
        }

        self.buffer_stock = {}
        self.station_progress = {}
        self.station_active_agent = {}
        self.station_recipe = {}
        self.station_queue = {}
        self.lane_open = {}
        self.lane_controller = {}
        self.sink_demand = {}

        for nd in self.nodes:
            if nd.kind == "buffer":
                self.buffer_stock[nd.node_id] = 0
            elif nd.kind == "lane":
                self.lane_open[nd.node_id] = 1
                self.lane_controller[nd.node_id] = -1
            elif nd.kind == "station":
                self.station_progress[nd.node_id] = 0.0
                self.station_active_agent[nd.node_id] = -1
                self.station_queue[nd.node_id] = []
                self.station_recipe[nd.node_id] = (nd.zone % 3, nd.zone % 3)
            elif nd.kind == "sink":
                self.sink_demand[nd.node_id] = nd.zone % 3

        if sample_hidden_rules:
            self._sample_hidden_institutional_rules()
        else:
            for node_id, controller in prev_lane_controller.items():
                if node_id in self.lane_controller:
                    self.lane_controller[node_id] = int(controller)

            for node_id, recipe in prev_station_recipe.items():
                if node_id in self.station_recipe:
                    self.station_recipe[node_id] = tuple(recipe)

            for node_id, demand in prev_sink_demand.items():
                if node_id in self.sink_demand:
                    self.sink_demand[node_id] = int(demand)

        self._compute_diagnostic_core_population()

    def _sample_hidden_institutional_rules(self):
        for nd in self.nodes:
            if nd.kind == "lane":
                nd.hidden_priority = int(self.rng.randint(0, self.n_agents))
                self.lane_controller[nd.node_id] = int(nd.hidden_priority)
                self.lane_open[nd.node_id] = 1

            elif nd.kind == "station":
                nd.hidden_degraded = int(self.rng.rand() < 0.35)
                raw_t = int(self.rng.randint(0, 3))
                prod_t = int(self.rng.randint(0, 3))
                self.station_recipe[nd.node_id] = (raw_t, prod_t)

            elif nd.kind == "sink":
                self.sink_demand[nd.node_id] = int(self.rng.randint(0, 3))

    def _behavioral_drift_update(self):
        pool = list(self.STYLES)

        for a in range(self.n_agents):
            if (a + self.current_phase) % 2 == 0:
                self.agent_style[a] = pool[(a + self.current_phase) % len(pool)]

    def _structural_shift_update(self):
        self._sample_hidden_institutional_rules()
        self._compute_diagnostic_core_population()

    def reset(self):
        phase_boundary = (
            self.episode_count > 0
            and self.episode_count % self.phase_length == 0
        )
        first_reset = self.episode_count == 0

        if phase_boundary:
            self.current_phase += 1

        self.step_count = 0

        self._assign_agents(
            force=(first_reset or self.resample_agent_layout_each_reset)
        )
        self._init_dynamic_state(
            sample_hidden_rules=(
                first_reset or self.resample_hidden_rules_each_reset
            )
        )

        if self.mode == "behavioral_drift":
            self._behavioral_drift_update()
        elif self.mode == "structural_shift" and phase_boundary:
            self._structural_shift_update()

        self._compute_diagnostic_core_population()

        self.episode_count += 1

        return self._get_obs_all()

    # =========================================================
    # Step
    # =========================================================

    def step(self, actions: List[int]):
        assert len(actions) == self.n_agents

        rewards = np.zeros(self.n_agents, dtype=np.float32)

        proposed = [
            self._propose_move(a, int(actions[a]))
            for a in range(self.n_agents)
        ]

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
        rewards += self._apply_local_mismatch_penalties(actions)
        rewards += self._throughput_shaping()

        self._update_recent_interactions(actions)
        self._update_pair_traces(actions)
        self._compute_diagnostic_core_population()

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "phase": int(self.current_phase),
            "diagnostic_core_by_ego": copy.deepcopy(self.diagnostic_core_by_ego),
        }

        return self._get_obs_all(), rewards.tolist(), bool(done), info

    # =========================================================
    # Mechanics
    # =========================================================

    def _propose_move(self, agent_id: int, action: int):
        r, c = self.positions[int(agent_id)]
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
        r, c = self.positions[int(agent_id)]
        out = []

        for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            pos = (r + dr, c + dc)

            if pos in self.node_by_pos:
                nd = self.nodes[self.node_by_pos[pos]]

                if kind is None or nd.kind == kind:
                    out.append(nd)

        return out

    def _handle_pick(self, agent_id: int) -> float:
        agent_id = int(agent_id)

        if self.inventory_qty[agent_id] > 0:
            return -0.02

        sources = self._adjacent_nodes(agent_id, "source")

        if not sources:
            return -0.02

        src = sources[0]
        carry_cap = 1 + int(round(float(self.agent_cap[agent_id][0])))

        self.inventory_type[agent_id] = int(src.resource_type)
        self.inventory_qty[agent_id] = min(1, carry_cap)

        return 0.03

    def _handle_drop_to_buffer(self, agent_id: int) -> float:
        agent_id = int(agent_id)

        if self.inventory_qty[agent_id] <= 0:
            return -0.02

        bufs = self._adjacent_nodes(agent_id, "buffer")

        if not bufs:
            return -0.02

        buf = bufs[0]
        st = self._get_zone_station(buf.zone)
        expected_raw, _ = self.station_recipe[st.node_id]

        self.buffer_stock[buf.node_id] = min(
            int(buf.capacity),
            int(self.buffer_stock[buf.node_id]) + int(self.inventory_qty[agent_id]),
        )

        rew = 0.08 if int(self.inventory_type[agent_id]) == int(expected_raw) else -0.08

        self.inventory_type[agent_id] = -1
        self.inventory_qty[agent_id] = 0

        return float(rew)

    def _handle_process(self, agent_id: int) -> float:
        agent_id = int(agent_id)

        stations = self._adjacent_nodes(agent_id, "station")

        if not stations:
            return -0.02

        st = stations[0]
        buf = self._get_zone_buffer(st.zone)
        lane = self._get_zone_lane(st.zone)

        privileged = int(self.lane_controller[lane.node_id]) == agent_id
        strong_lane_access = float(self.agent_cap[agent_id][2]) > 0.85

        if not self.lane_open[lane.node_id] and not privileged and not strong_lane_access:
            return -0.05

        if st.hidden_degraded and float(self.agent_cap[agent_id][1]) < 0.8:
            return -0.04

        if agent_id == int(self.station_active_agent[st.node_id]):
            return 0.0

        if agent_id in self.station_queue[st.node_id]:
            return 0.0

        if len(self.station_queue[st.node_id]) >= int(st.queue_limit):
            return -0.03

        if int(self.buffer_stock[buf.node_id]) <= 0:
            return -0.03

        if int(self.station_active_agent[st.node_id]) == -1:
            self.buffer_stock[buf.node_id] -= 1
            self.station_active_agent[st.node_id] = agent_id
            self.station_progress[st.node_id] = 0.0
            return 0.05

        self.station_queue[st.node_id].append(agent_id)

        return 0.01

    def _advance_station_processing(self) -> np.ndarray:
        rew = np.zeros(self.n_agents, dtype=np.float32)

        for nd in self.nodes:
            if nd.kind != "station":
                continue

            st_id = nd.node_id
            active = int(self.station_active_agent[st_id])

            if active == -1:
                continue

            speed = 1.0 + 0.5 * float(self.agent_cap[active][1])

            if nd.hidden_degraded:
                speed *= 0.6

            self.station_progress[st_id] += float(speed)

            if self.station_progress[st_id] >= 2.0:
                _, prod_t = self.station_recipe[st_id]

                if int(self.inventory_qty[active]) == 0:
                    self.inventory_type[active] = int(prod_t)
                    self.inventory_qty[active] = 1
                    rew[active] += 0.18 if not nd.hidden_degraded else 0.10
                else:
                    rew[active] -= 0.02

                if len(self.station_queue[st_id]) > 0:
                    nxt = int(self.station_queue[st_id].pop(0))
                    buf = self._get_zone_buffer(nd.zone)

                    if int(self.buffer_stock[buf.node_id]) > 0:
                        self.buffer_stock[buf.node_id] -= 1
                        self.station_active_agent[st_id] = nxt
                        self.station_progress[st_id] = 0.0
                    else:
                        self.station_active_agent[st_id] = -1
                        self.station_progress[st_id] = 0.0
                else:
                    self.station_active_agent[st_id] = -1
                    self.station_progress[st_id] = 0.0

        return rew

    def _handle_deliver(self, agent_id: int) -> float:
        agent_id = int(agent_id)

        if self.inventory_qty[agent_id] <= 0:
            return -0.02

        sinks = self._adjacent_nodes(agent_id, "sink")

        if not sinks:
            return -0.02

        sk = sinks[0]
        demanded = int(self.sink_demand[sk.node_id])

        rew = 1.2 if int(self.inventory_type[agent_id]) == demanded else -0.25

        self.inventory_type[agent_id] = -1
        self.inventory_qty[agent_id] = 0

        return float(rew)

    def _handle_toggle_lane(self, agent_id: int) -> float:
        agent_id = int(agent_id)

        lanes = self._adjacent_nodes(agent_id, "lane")

        if not lanes:
            return -0.01

        lane = lanes[0]

        privileged = int(self.lane_controller[lane.node_id]) == agent_id
        strong_access = float(self.agent_cap[agent_id][2]) > 0.85

        if privileged:
            self.lane_open[lane.node_id] = 1 - int(self.lane_open[lane.node_id])
            return 0.05

        if strong_access:
            self.lane_open[lane.node_id] = 1 - int(self.lane_open[lane.node_id])
            return 0.02

        return -0.03

    def _handle_inspect(self, agent_id: int) -> float:
        agent_id = int(agent_id)

        nearby = self._adjacent_nodes(agent_id, None)

        if not nearby:
            return -0.01

        rew = 0.0

        for nd in nearby:
            if nd.kind == "lane":
                self.inspect_memory[agent_id][f"lane_controller_zone_{nd.zone}"] = (
                    float(self.lane_controller[nd.node_id]) / max(1, self.n_agents - 1)
                )
                rew += 0.01

            elif nd.kind == "station":
                self.inspect_memory[agent_id][f"station_degraded_zone_{nd.zone}"] = float(nd.hidden_degraded)
                raw_t, prod_t = self.station_recipe[nd.node_id]
                self.inspect_memory[agent_id][f"station_raw_zone_{nd.zone}"] = float(raw_t) / 2.0
                self.inspect_memory[agent_id][f"station_prod_zone_{nd.zone}"] = float(prod_t) / 2.0
                rew += 0.01

            elif nd.kind == "sink":
                self.inspect_memory[agent_id][f"sink_demand_zone_{nd.zone}"] = (
                    float(self.sink_demand[nd.node_id]) / 2.0
                )
                rew += 0.01

        return float(rew)

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
                controller = int(self.lane_controller[lane.node_id])

                for a in ids:
                    rew[a] -= 0.01 if int(a) == controller else 0.08
            else:
                for a in ids:
                    rew[a] -= 0.03

        return rew

    def _apply_local_mismatch_penalties(self, actions: List[int]) -> np.ndarray:
        rew = np.zeros(self.n_agents, dtype=np.float32)

        for a, act in enumerate(actions):
            act = int(act)

            if act == self.DROP_TO_BUFFER:
                bufs = self._adjacent_nodes(a, "buffer")

                if bufs:
                    st = self._get_zone_station(bufs[0].zone)
                    raw_t, _ = self.station_recipe[st.node_id]

                    if self.inventory_type[a] != -1 and int(self.inventory_type[a]) != int(raw_t):
                        rew[a] -= 0.03

            if act == self.PROCESS:
                stations = self._adjacent_nodes(a, "station")

                if stations and stations[0].hidden_degraded and float(self.agent_cap[a][1]) < 0.8:
                    rew[a] -= 0.01

        return rew

    def _throughput_shaping(self) -> np.ndarray:
        rew = np.zeros(self.n_agents, dtype=np.float32)

        for z in range(self.n_zones):
            buf = self._get_zone_buffer(z)
            lane = self._get_zone_lane(z)
            st = self._get_zone_station(z)
            sk = self._get_zone_sink(z)

            if int(self.buffer_stock[buf.node_id]) > 0 and int(self.lane_open[lane.node_id]) == 1:
                for a in range(self.n_agents):
                    if int(self.agent_zone[a]) == z:
                        rew[a] += 0.002

            if st.hidden_degraded:
                for a in range(self.n_agents):
                    if int(self.agent_zone[a]) == z:
                        rew[a] -= 0.001

            sink_near = self.positions_near(sk.pos, radius=1)
            foreign = 0

            for a in range(self.n_agents):
                if tuple(self.positions[a]) in sink_near and int(self.agent_zone[a]) != z:
                    foreign += 1

            if foreign > 0:
                for a in range(self.n_agents):
                    if int(self.agent_zone[a]) == z:
                        rew[a] -= 0.002 * float(foreign)

        return rew

    # =========================================================
    # Histories / pair traces
    # =========================================================

    def _update_recent_interactions(self, actions: List[int]):
        for ego in range(self.n_agents):
            er, ec = self.positions[ego]
            local = []

            for j in range(self.n_agents):
                if j == ego:
                    continue

                jr, jc = self.positions[j]
                dist = abs(er - jr) + abs(ec - jc)

                if dist <= 2:
                    local.append((int(j), int(actions[j]), int(self.last_signals[j])))

            self.recent_interactions[ego].extend(local)
            self.recent_interactions[ego] = self.recent_interactions[ego][-16:]

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

                if int(self.last_signals[j]) > 0:
                    tr[3] += 1.0

                if pi == pj and pi in lane_positions:
                    controller = self._lane_controller_at_pos(pi)

                    if controller == j or float(self.agent_cap[j][2]) > float(self.agent_cap[i][2]):
                        tr[4] += 1.0

                if pi in sink_positions and pj in sink_positions and int(self.agent_zone[i]) != int(self.agent_zone[j]):
                    tr[5] += 1.0

    # =========================================================
    # Diagnostic structural core
    # =========================================================

    def _diagnostic_core_budget(self):
        n_neighbors = max(1, int(self.n_agents) - 1)

        if hasattr(self, "diagnostic_core_k"):
            try:
                return int(max(1, min(int(self.diagnostic_core_k), n_neighbors)))
            except Exception:
                pass

        if hasattr(self, "max_core_size"):
            try:
                return int(max(1, min(int(self.max_core_size), n_neighbors)))
            except Exception:
                pass

        if n_neighbors <= 4:
            return min(2, n_neighbors)

        if n_neighbors <= 12:
            return min(3, n_neighbors)

        return min(4, n_neighbors)

    def _role_structural_bonus(self, role):
        role = str(role)

        bonuses = {
            "controller": 0.35,
            "dispatcher": 0.32,
            "processor": 0.30,
            "hauler": 0.22,
            "collector": 0.18,
            "signaler": 0.18,
            "sweeper": 0.10,
            "spoiler": 0.08,
        }

        return float(bonuses.get(role, 0.0))

    def _pair_trace_score(self, ego, j):
        ego = int(ego)
        j = int(j)
        pair = (ego, j)

        if not hasattr(self, "pair_trace_counts"):
            return 0.0

        if pair not in self.pair_trace_counts:
            return 0.0

        tr = np.asarray(self.pair_trace_counts[pair], dtype=np.float32)

        if tr.size == 0:
            return 0.0

        weights = np.asarray([0.40, 0.55, 0.45, 0.35, 0.80, 0.55], dtype=np.float32)
        denom = max(1.0, float(self.step_count + 1))

        return float(np.dot(tr / denom, weights))

    def _same_local_resource_chain(self, ego, j):
        ego = int(ego)
        j = int(j)

        try:
            zi = int(self.agent_zone[ego])
            zj = int(self.agent_zone[j])

            same_zone = 1.0 if zi == zj else 0.0
            near_zone = 1.0 if abs(zi - zj) <= 1 else 0.0
        except Exception:
            same_zone = 0.0
            near_zone = 0.0

        return float(0.25 * same_zone + 0.08 * near_zone)

    def _diagnostic_pair_score(self, ego, j):
        ego = int(ego)
        j = int(j)

        if ego == j:
            return -1e9

        pi = self.positions[ego]
        pj = self.positions[j]

        grid_den = max(1, int(getattr(self, "grid_size", 1)))
        dist = abs(pj[0] - pi[0]) + abs(pj[1] - pi[1])

        proximity = 1.0 / (1.0 + float(dist))
        local_chain = self._same_local_resource_chain(ego, j)

        role_bonus = 0.0

        if hasattr(self, "agent_role"):
            try:
                role_bonus = self._role_structural_bonus(self.agent_role[j])
            except Exception:
                role_bonus = 0.0

        trace_raw = self._pair_trace_score(ego, j)
        trace_score = float(np.tanh(trace_raw))

        controller_bonus = 0.0

        if hasattr(self, "lane_controller"):
            try:
                if int(j) in [int(v) for v in self.lane_controller.values()]:
                    controller_bonus += 0.25
            except Exception:
                pass

        signal_bonus = 0.0

        if hasattr(self, "last_signals"):
            try:
                if int(self.last_signals[j]) > 0:
                    signal_bonus += 0.08
            except Exception:
                pass

        same_cell_bonus = 0.0

        try:
            if tuple(self.positions[ego]) == tuple(self.positions[j]):
                same_cell_bonus += 0.25
        except Exception:
            pass

        score = (
            0.35 * proximity
            + local_chain
            + role_bonus
            + 0.45 * trace_score
            + controller_bonus
            + signal_bonus
            + same_cell_bonus
        )

        return float(score)

    def _compute_diagnostic_core_population(self):
        out = {}
        k = int(self._diagnostic_core_budget())

        for ego in range(int(self.n_agents)):
            scored = []

            for j in range(int(self.n_agents)):
                if j == ego:
                    continue

                score = self._diagnostic_pair_score(ego, j)
                scored.append((int(j), float(score)))

            scored.sort(key=lambda x: x[1], reverse=True)

            chosen = [j for j, _ in scored[:k]]
            out[int(ego)] = set(chosen)

        self.diagnostic_core_by_ego = out

        return out

    # =========================================================
    # Oracle intervention hooks
    # =========================================================

    def scripted_policy(self, agent_id: int) -> int:
        agent_id = int(agent_id)

        style = self.agent_style[agent_id]
        zone = int(self.agent_zone[agent_id])
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
            if (
                self.inventory_qty[agent_id] == 0
                and self._manhattan(pos, st.pos) <= 1
                and self.buffer_stock[buf.node_id] > 0
            ):
                return self.PROCESS
            if self.inventory_qty[agent_id] > 0 and self._manhattan(pos, sk.pos) <= 1:
                return self.DELIVER
            return self._greedy_move_toward(
                agent_id,
                src.pos if self.inventory_qty[agent_id] == 0 else buf.pos,
            )

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
                return self.SIGNAL_B if self.current_phase % 2 == 0 else self.SIGNAL_A
            return self._greedy_move_toward(agent_id, lane.pos)

        if style == "conservative":
            if self._manhattan(pos, st.pos) <= 1:
                return self.INSPECT
            return self.STAY if self.rng.rand() < 0.4 else self._greedy_move_toward(agent_id, st.pos)

        if style == "retaliatory":
            nearby = self._visible_agents_for(agent_id)

            if any(self.last_signals[j] == 2 for j in nearby if j != agent_id):
                return self.TOGGLE_LANE

            return self._greedy_move_toward(agent_id, lane.pos)

        return self.STAY

    def default_joint_action(self) -> List[int]:
        return [self.scripted_policy(a) for a in range(self.n_agents)]

    def rollout_from_current_state(
        self,
        forced: Optional[Dict[int, int]] = None,
        horizon: int = 3,
        discount: float = 0.95,
    ) -> np.ndarray:
        saved = self.clone_state()
        out = np.zeros(self.n_agents, dtype=np.float32)

        forced = forced or {}

        for h in range(int(horizon)):
            acts = self.default_joint_action()

            for aid, act in forced.items():
                acts[int(aid)] = int(act)

            _, rewards, done, _ = self.step(acts)
            out += (float(discount) ** h) * np.asarray(rewards, dtype=np.float32)

            if done:
                break

        self.restore_state(saved)

        return out

    def compute_oracle_influence_from_current_state(
        self,
        ego_id: int,
        agent_j: Optional[int] = None,
        intervention_action: Optional[int] = None,
        horizon: int = 3,
        n_trials: int = 3,
        discount: float = 0.95,
        neighbor_id: Optional[int] = None,
    ) -> float:
        if agent_j is None:
            agent_j = neighbor_id

        if agent_j is None:
            raise ValueError("compute_oracle_influence_from_current_state requires agent_j or neighbor_id.")

        ego_id = int(ego_id)
        agent_j = int(agent_j)

        if intervention_action is None:
            candidate_actions = [
                self.STAY,
                self.SIGNAL_A,
                self.SIGNAL_B,
                self.TOGGLE_LANE,
            ]
        else:
            candidate_actions = [int(intervention_action)]

        saved = self.clone_state()
        deltas = []

        for action in candidate_actions:
            for _ in range(int(max(1, n_trials))):
                self.restore_state(saved)
                base_returns = self.rollout_from_current_state(
                    forced=None,
                    horizon=int(horizon),
                    discount=float(discount),
                )

                self.restore_state(saved)
                int_returns = self.rollout_from_current_state(
                    forced={agent_j: int(action)},
                    horizon=int(horizon),
                    discount=float(discount),
                )

                deltas.append(float(int_returns[ego_id] - base_returns[ego_id]))

        self.restore_state(saved)

        if len(deltas) == 0:
            return 0.0

        return float(np.mean(deltas))

    def estimate_oracle_core_from_current_state(
        self,
        ego_id: int,
        horizon: int = 3,
        n_trials: int = 2,
        top_k: int = 3,
        discount: float = 0.95,
    ) -> Tuple[Dict[int, float], List[int]]:
        ego_id = int(ego_id)

        scores = {}

        candidate_actions = [
            self.STAY,
            self.SIGNAL_A,
            self.SIGNAL_B,
            self.TOGGLE_LANE,
        ]

        for j in range(self.n_agents):
            if j == ego_id:
                continue

            vals = []

            for a in candidate_actions:
                vals.append(
                    abs(
                        self.compute_oracle_influence_from_current_state(
                            ego_id=ego_id,
                            agent_j=j,
                            intervention_action=a,
                            horizon=horizon,
                            n_trials=n_trials,
                            discount=discount,
                        )
                    )
                )

            scores[int(j)] = float(np.mean(vals))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = [j for j, _ in ranked[: int(top_k)]]

        return scores, top

    def sample_state_bank(self, n_states: int = 8, burn_in: int = 4):
        state_bank = []
        self.reset()

        for _ in range(int(max(0, burn_in))):
            actions = self.default_joint_action()
            _, _, done, _ = self.step(actions)

            if done:
                self.reset()

        for _ in range(int(max(1, n_states))):
            state_bank.append(self.clone_state())

            actions = self.default_joint_action()
            _, _, done, _ = self.step(actions)

            if done:
                self.reset()

        return state_bank

    def get_supported_egos(self):
        return list(range(self.n_agents))

    # =========================================================
    # Observation
    # =========================================================

    def _get_obs_all(self):
        return [self._get_obs(a) for a in range(self.n_agents)]

    def get_all_obs(self):
        return self._get_obs_all()

    def get_obs_of_ego(self, obs_all, ego_id: int):
        ego_id = int(ego_id)

        if obs_all is None:
            return self._get_obs(ego_id)

        return obs_all[ego_id]

    def _get_obs(self, agent_id: int):
        agent_id = int(agent_id)

        r0, c0 = self.positions[agent_id]
        sr = self.obs_radius

        patch = []

        for dr in range(-sr, sr + 1):
            for dc in range(-sr, sr + 1):
                rr, cc = r0 + dr, c0 + dc
                patch.append(float(self._tile_code(rr, cc)) / 7.0)

        visible_agents = [
            j
            for j in self._visible_agents_for(agent_id)
            if j != agent_id
        ]

        visible_agents = sorted(
            visible_agents,
            key=lambda j: abs(self.positions[j][0] - r0) + abs(self.positions[j][1] - c0),
        )[: self.max_visible_agents]

        nearby_feats = []

        for j in visible_agents:
            rr = (self.positions[j][0] - r0) / max(1, self.grid_size)
            cc = (self.positions[j][1] - c0) / max(1, self.grid_size)

            nearby_feats.extend(
                [
                    float(rr),
                    float(cc),
                    float(self.agent_zone[j]) / max(1, self.n_zones - 1),
                    float(self.inventory_type[j] + 1) / 3.0,
                    float(self.last_signals[j]) / 2.0,
                    float(self.last_actions[j]) / max(1, self.get_action_dim() - 1),
                ]
            )

        while len(nearby_feats) < self.max_visible_agents * 6:
            nearby_feats.extend([0.0] * 6)

        public = []

        for z in range(self.n_zones):
            buf = self._get_zone_buffer(z)
            st = self._get_zone_station(z)
            sk = self._get_zone_sink(z)
            lane = self._get_zone_lane(z)

            public.extend(
                [
                    float(self.buffer_stock[buf.node_id]) / max(1, buf.capacity),
                    float(len(self.station_queue[st.node_id])) / max(1, st.queue_limit),
                    float(self.station_active_agent[st.node_id] != -1),
                    float(self.lane_open[lane.node_id]),
                    float(self.sink_demand[sk.node_id]) / 2.0,
                ]
            )

        private = []
        private.extend(self.agent_cap[agent_id].tolist())
        private.extend(
            [
                float(self.inventory_type[agent_id] + 1) / 3.0,
                float(self.inventory_qty[agent_id]),
                float(self.agent_zone[agent_id]) / max(1, self.n_zones - 1),
                float(self.current_phase) / 10.0,
            ]
        )

        mem = []
        keys = []

        for z in range(self.n_zones):
            keys.extend(
                [
                    f"lane_controller_zone_{z}",
                    f"station_degraded_zone_{z}",
                    f"station_raw_zone_{z}",
                    f"station_prod_zone_{z}",
                    f"sink_demand_zone_{z}",
                ]
            )

        for k in keys:
            mem.append(float(self.inspect_memory[agent_id].get(k, -1.0)))

        pair_feats = []

        candidates = sorted(
            [j for j in range(self.n_agents) if j != agent_id],
            key=lambda j: abs(self.positions[j][0] - r0) + abs(self.positions[j][1] - c0),
        )[: self.max_pair_trace_neighbors]

        for j in candidates:
            tr = self.pair_trace_counts[(agent_id, j)]
            denom = max(1, self.step_count + 1)
            pair_feats.extend((tr / float(denom)).tolist())

        while len(pair_feats) < self.max_pair_trace_neighbors * 6:
            pair_feats.extend([0.0] * 6)

        obs = np.asarray(
            patch + nearby_feats + public + private + mem + pair_feats,
            dtype=np.float32,
        )

        return obs

    def _visible_agents_for(self, agent_id: int):
        agent_id = int(agent_id)

        r0, c0 = self.positions[agent_id]
        out = []

        for j in range(self.n_agents):
            r1, c1 = self.positions[j]

            if max(abs(r1 - r0), abs(c1 - c0)) > self.obs_radius:
                continue

            hidden = False

            if (r1, c1) in self.occluders or (r0, c0) in self.occluders:
                hidden = True

            if hidden and float(self.agent_cap[agent_id][3]) < 0.95 and agent_id != j:
                continue

            out.append(j)

        return out

    def _tile_code(self, r: int, c: int):
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return self.TILE_WALL

        pos = (int(r), int(c))

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

    def _get_zone_source(self, zone: int) -> Node:
        return next(nd for nd in self.nodes if nd.zone == int(zone) and nd.kind == "source")

    def _get_zone_buffer(self, zone: int) -> Node:
        return next(nd for nd in self.nodes if nd.zone == int(zone) and nd.kind == "buffer")

    def _get_zone_station(self, zone: int) -> Node:
        return next(nd for nd in self.nodes if nd.zone == int(zone) and nd.kind == "station")

    def _get_zone_sink(self, zone: int) -> Node:
        return next(nd for nd in self.nodes if nd.zone == int(zone) and nd.kind == "sink")

    def _get_zone_lane(self, zone: int) -> Node:
        return next(nd for nd in self.nodes if nd.zone == int(zone) and nd.kind == "lane")

    def _lane_controller_at_pos(self, pos: Tuple[int, int]) -> int:
        pos = tuple(pos)

        if pos not in self.node_by_pos:
            return -1

        nd = self.nodes[self.node_by_pos[pos]]

        if nd.kind != "lane":
            return -1

        return int(self.lane_controller[nd.node_id])

    def positions_near(self, pos: Tuple[int, int], radius: int = 1):
        out = set()
        r0, c0 = pos

        for r in range(max(1, r0 - radius), min(self.grid_size - 1, r0 + radius + 1)):
            for c in range(max(1, c0 - radius), min(self.grid_size - 1, c0 + radius + 1)):
                out.add((r, c))

        return out

    def _manhattan(self, p1, p2):
        return int(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))

    def _greedy_move_toward(self, agent_id: int, target: Tuple[int, int]) -> int:
        agent_id = int(agent_id)

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
    # Dimensions
    # =========================================================

    def get_obs_dim(self):
        patch_dim = (2 * self.obs_radius + 1) ** 2
        nearby_dim = self.max_visible_agents * 6
        public_dim = self.n_zones * 5
        private_dim = 8
        mem_dim = self.n_zones * 5
        pair_dim = self.max_pair_trace_neighbors * 6

        return int(patch_dim + nearby_dim + public_dim + private_dim + mem_dim + pair_dim)

    def get_action_dim(self):
        return 13

    # =========================================================
    # Clone / restore
    # =========================================================

    def clone_state(self):
        return copy.deepcopy(
            {
                "episode_count": self.episode_count,
                "current_phase": self.current_phase,
                "step_count": self.step_count,
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
                "station_recipe": self.station_recipe,
                "station_queue": self.station_queue,
                "lane_open": self.lane_open,
                "lane_controller": self.lane_controller,
                "sink_demand": self.sink_demand,
                "diagnostic_core_by_ego": self.diagnostic_core_by_ego,
                "recent_interactions": self.recent_interactions,
                "pair_trace_counts": self.pair_trace_counts,
                "nodes": self.nodes,
                "resample_agent_layout_each_reset": self.resample_agent_layout_each_reset,
                "resample_hidden_rules_each_reset": self.resample_hidden_rules_each_reset,
                "rng_state": self.rng.get_state(),
            }
        )

    def restore_state(self, state):
        self.episode_count = copy.deepcopy(state["episode_count"])
        self.current_phase = copy.deepcopy(state["current_phase"])
        self.step_count = copy.deepcopy(state["step_count"])
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
        self.station_recipe = copy.deepcopy(state["station_recipe"])
        self.station_queue = copy.deepcopy(state["station_queue"])
        self.lane_open = copy.deepcopy(state["lane_open"])
        self.lane_controller = copy.deepcopy(state["lane_controller"])
        self.sink_demand = copy.deepcopy(state["sink_demand"])
        self.diagnostic_core_by_ego = copy.deepcopy(state["diagnostic_core_by_ego"])
        self.recent_interactions = copy.deepcopy(state["recent_interactions"])
        self.pair_trace_counts = copy.deepcopy(state["pair_trace_counts"])
        self.nodes = copy.deepcopy(state["nodes"])
        self.resample_agent_layout_each_reset = copy.deepcopy(
            state.get(
                "resample_agent_layout_each_reset",
                getattr(self, "resample_agent_layout_each_reset", False),
            )
        )
        self.resample_hidden_rules_each_reset = copy.deepcopy(
            state.get(
                "resample_hidden_rules_each_reset",
                getattr(self, "resample_hidden_rules_each_reset", False),
            )
        )
        self.rng.set_state(state["rng_state"])
