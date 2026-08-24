"""CityFlow traffic-light population adapter for CIG-AMF."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from envs.external_contract import BenchmarkCapabilities
from envs.external.common import ExternalPopulationMixin


class CityFlowCIGEnvironment(ExternalPopulationMixin):
    capabilities = BenchmarkCapabilities("CityFlow", True, True, True, True, False, False, False)

    def __init__(self, engine, roadnet_path, max_steps=60, observation_width=64):
        self.raw_env = engine
        self.max_steps = int(max_steps)
        self.obs_dim = int(observation_width)
        self._step_count = 0
        roadnet = json.loads(Path(roadnet_path).read_text(encoding="utf-8"))
        intersections = [x for x in roadnet.get("intersections", []) if not x.get("virtual", False)]
        self.agent_ids = [str(x["id"]) for x in intersections]
        if not self.agent_ids:
            raise RuntimeError("CityFlow roadnet has no controllable intersections")
        self._intersection = {str(x["id"]): x for x in intersections}
        self._phase_counts = {}
        for iid, item in self._intersection.items():
            phases = item.get("trafficLight", {}).get("lightphases", [])
            self._phase_counts[iid] = max(1, len(phases))
        self.n_agents = len(self.agent_ids)
        self.max_action_dim = max(self._phase_counts.values())
        self._adj = self._build_adjacency(roadnet)
        self._obs = [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.n_agents)]
        self._finish_init()

    def _build_adjacency(self, roadnet):
        adj = {iid: set() for iid in self.agent_ids}
        for road in roadnet.get("roads", []):
            a, b = str(road.get("startIntersection")), str(road.get("endIntersection"))
            if a in adj and b in adj:
                adj[a].add(b); adj[b].add(a)
        return adj

    def _refresh_obs(self):
        lane_count = self.raw_env.get_lane_vehicle_count()
        waiting = self.raw_env.get_lane_waiting_vehicle_count()
        rows = []
        for iid in self.agent_ids:
            item = self._intersection[iid]
            road_ids = set(str(x) for x in item.get("roads", []))
            selected_lanes = [lane for lane in lane_count if any(lane.startswith(r) for r in road_ids)]
            counts = [float(lane_count.get(lane, 0.0)) for lane in selected_lanes]
            waits = [float(waiting.get(lane, 0.0)) for lane in selected_lanes]
            features = np.asarray([
                sum(counts), sum(waits), np.mean(counts) if counts else 0.0,
                np.mean(waits) if waits else 0.0, float(len(self._adj.get(iid, ()))),
                float(self.raw_env.get_current_time()) / max(1.0, float(self.max_steps)),
            ], dtype=np.float32)
            out = np.zeros((self.obs_dim,), dtype=np.float32)
            out[:features.size] = features
            rows.append(out)
        self._obs = rows

    def reset(self, seed=None):
        if seed is not None and hasattr(self.raw_env, "set_random_seed"):
            self.raw_env.set_random_seed(int(seed))
        self.raw_env.reset(seed=bool(seed is not None))
        self._step_count = 0
        self.last_actions = [0] * self.n_agents
        self._refresh_obs()
        return self._obs

    def step(self, actions):
        applied_actions = []
        for idx, iid in enumerate(self.agent_ids):
            mask = self.valid_action_mask(idx)
            action = int(actions[idx])
            if not (0 <= action < mask.size and mask[action]):
                action = int(np.flatnonzero(mask)[0])
            applied_actions.append(action)
            self.raw_env.set_tl_phase(iid, action)
        self.raw_env.next_step()
        self._step_count += 1
        waiting = self.raw_env.get_lane_waiting_vehicle_count()
        rewards = []
        for iid in self.agent_ids:
            roads = set(str(x) for x in self._intersection[iid].get("roads", []))
            local = [float(v) for lane, v in waiting.items() if any(lane.startswith(r) for r in roads)]
            rewards.append(-float(sum(local)))
        self.last_actions = applied_actions
        self._refresh_obs()
        return self._obs, rewards, bool(self._step_count >= self.max_steps), {}

    def valid_action_mask(self, agent):
        iid = self.agent_ids[int(agent)]
        mask = np.zeros((self.max_action_dim,), dtype=bool)
        mask[: self._phase_counts[iid]] = True
        return mask

    def relation_features(self, ego, neighbour):
        a, b = self.agent_ids[int(ego)], self.agent_ids[int(neighbour)]
        adjacent = float(b in self._adj.get(a, set()))
        deg_a = float(len(self._adj.get(a, ())))
        deg_b = float(len(self._adj.get(b, ())))
        oi, oj = self._obs[int(ego)], self._obs[int(neighbour)]
        load_diff = float((oj[0] - oi[0]) / (1.0 + abs(oi[0]) + abs(oj[0])))
        wait_diff = float((oj[1] - oi[1]) / (1.0 + abs(oi[1]) + abs(oj[1])))
        phase_ratio = float(self._phase_counts[b] / max(1, self._phase_counts[a]))
        common = float(len(self._adj.get(a, set()).intersection(self._adj.get(b, set()))))
        return np.asarray([adjacent, deg_a / 8.0, deg_b / 8.0, load_diff, wait_diff, phase_ratio + 0.01 * common], dtype=np.float32)

    def clone_state(self):
        return (self.raw_env.snapshot(), [x.copy() for x in self._obs], list(self.last_actions), self._step_count, self._behaviour_override)

    def restore_state(self, state):
        archive, obs, actions, step_count, behaviour = state
        self.raw_env.load(archive)
        self._obs = [x.copy() for x in obs]
        self.last_actions = list(actions)
        self._step_count = int(step_count)
        self._behaviour_override = str(behaviour)

    def _copy_cloned_state(self, state):
        archive, obs, actions, step_count, behaviour = state
        return (archive, [x.copy() for x in obs], list(actions), int(step_count), str(behaviour))

    def fixed_continuation_policy(self, agent):
        # Hold the currently commanded signal phase under rho.
        action = int(self.last_actions[int(agent)])
        mask = self.valid_action_mask(agent)
        return action if 0 <= action < mask.size and mask[action] else int(np.flatnonzero(mask)[0])


def make_cityflow_environment(seed=0, repo_path=None, config_path=None, max_steps=60, observation_width=64, **_):
    import os
    import tempfile
    import cityflow
    if config_path is None:
        if repo_path is None:
            raise RuntimeError("CityFlow requires repo_path or --config-path")
        preferred = Path(repo_path) / "examples" / "config.json"
        if preferred.exists():
            config_path = preferred
        else:
            candidates = sorted(Path(repo_path).glob("examples/**/*.json"))
            for candidate in candidates:
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    continue
                if "roadnetFile" in payload and "flowFile" in payload:
                    config_path = candidate
                    break
        if config_path is None:
            raise RuntimeError("could not locate a runnable CityFlow example config")
    config_path = Path(config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    dir_value = Path(str(payload.get("dir", ".")))
    if dir_value.is_absolute():
        data_root = dir_value
    else:
        roots = []
        if repo_path is not None:
            roots.append(Path(repo_path).resolve() / dir_value)
        roots.extend([config_path.parent / dir_value, config_path.parent])
        data_root = None
        for root in roots:
            if (root / payload["roadnetFile"]).exists() and (root / payload["flowFile"]).exists():
                data_root = root.resolve()
                break
        if data_root is None:
            raise RuntimeError(
                "CityFlow config paths do not resolve to existing roadnet/flow files: "
                f"config={config_path} dir={payload.get('dir')}"
            )
    roadnet_path = (data_root / payload["roadnetFile"]).resolve()

    # Upstream examples ship with rlTrafficLight=false, which makes
    # set_tl_phase() a no-op. Never mutate the pinned repository: create a
    # temporary control-enabled config with replay disabled.
    runtime_payload = dict(payload)
    runtime_payload["dir"] = str(data_root) + os.sep
    runtime_payload["rlTrafficLight"] = True
    runtime_payload["saveReplay"] = False
    runtime_payload.pop("roadnetLogFile", None)
    runtime_payload.pop("replayLogFile", None)
    fd, runtime_config = tempfile.mkstemp(prefix="cig-cityflow-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(runtime_payload, handle)
        engine = cityflow.Engine(runtime_config, thread_num=1)
    finally:
        try:
            os.unlink(runtime_config)
        except OSError:
            pass
    if hasattr(engine, "set_random_seed"):
        engine.set_random_seed(int(seed))
    env = CityFlowCIGEnvironment(engine, roadnet_path, max_steps=max_steps, observation_width=observation_width)
    env.reset(seed=int(seed))
    return env
