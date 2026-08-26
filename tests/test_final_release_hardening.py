"""Release-candidate regressions for paper/runtime wiring found in the v19 audit."""
from __future__ import annotations

import copy
import json
import inspect
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


class _RailEnvCloneFromDouble:
    number_of_agents = 2

    def __init__(self):
        self.agents = [object(), object()]
        self.payload = {"step": 7, "nested": [1, 2, 3]}
        self.clone_from_calls = 0

    def clone_from(self, env):
        self.clone_from_calls += 1
        self.payload = copy.deepcopy(env.payload)
        return None


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

    def oracle_lag_response(self, ego_id, agent_j, intervention_action, horizon=2, **kwargs):
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
        return obs, [0.0] * self.n_agents, False, {}


class _CityCollisionEngine:
    def __init__(self):
        self.t = 0
        self.phases = {}

    def get_lane_vehicle_count(self):
        return {"road_1_0": 2, "road_10_0": 100}

    def get_lane_waiting_vehicle_count(self):
        return {"road_1_0": 3, "road_10_0": 200}

    def get_current_time(self):
        return float(self.t)

    def set_random_seed(self, seed):
        self.seed = seed

    def reset(self, seed=False):
        self.t = 0
        self.phases = {}

    def set_tl_phase(self, iid, action):
        self.phases[iid] = int(action)

    def next_step(self):
        self.t += 1

    def snapshot(self):
        return copy.deepcopy((self.t, self.phases))

    def load(self, state):
        self.t, self.phases = copy.deepcopy(state)


class _Discrete:
    def __init__(self, n):
        self.n = int(n)


class _CyborgNeverDone:
    possible_agents = ["blue_agent_0", "blue_agent_1"]
    agents = possible_agents

    def __init__(self):
        self.t = 0

    def action_space(self, name):
        return _Discrete(3)

    def reset(self, seed=None):
        self.t = 0
        return {name: np.asarray([idx, 0.0]) for idx, name in enumerate(self.possible_agents)}

    def step(self, actions):
        self.t += 1
        obs = {name: np.asarray([idx, self.t]) for idx, name in enumerate(self.possible_agents)}
        rewards = {name: 0.0 for name in self.possible_agents}
        dones = {name: False for name in self.possible_agents}
        return obs, rewards, dones, {}


class _FlatlandNeverDone:
    number_of_agents = 1

    def __init__(self):
        self.agents = [object()]
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return {0: np.asarray([0.0, 0.0], dtype=np.float32)}, {}

    def step(self, actions):
        self.t += 1
        return (
            {0: np.asarray([0.0, self.t], dtype=np.float32)},
            {0: 0.0},
            {0: False, "__all__": False},
            {},
        )


class FinalReleaseHardeningTests(unittest.TestCase):
    def test_flatland_clone_uses_destination_clone_from_contract(self):
        from envs.flatland_adapter import FlatlandCIGEnvironment

        raw = _RailEnvCloneFromDouble()
        env = FlatlandCIGEnvironment(raw, observation_width=4)
        env._obs = [np.ones(4, dtype=np.float32) for _ in range(2)]
        env._step_count = 5
        state = env.clone_state()
        cloned_raw = state[0]
        self.assertIsNot(cloned_raw, raw)
        self.assertEqual(cloned_raw.clone_from_calls, 1)
        self.assertEqual(cloned_raw.payload, raw.payload)
        raw.payload["nested"].append(99)
        self.assertNotEqual(cloned_raw.payload, raw.payload)
        env._step_count = 99
        env.restore_state(state)
        self.assertEqual(env._step_count, 5)

    def test_flatland_action_mask_safe_stop_and_invalid_topology_fail_closed(self):
        from envs.flatland_adapter import FlatlandCIGEnvironment

        class State:
            def is_off_map_state(self):
                return False

        class Speed:
            max_speed = 1

            def is_cell_exit(self, speed):
                return True

        class Rail:
            def __init__(self, transitions):
                self.transitions = transitions

            def apply_action_independent(self, action, configuration):
                return None

            def get_transitions(self, configuration):
                return self.transitions

        agent = SimpleNamespace(
            current_configuration=((4, 4), 1),
            initial_configuration=((4, 4), 1),
            target_configuration=None,
            state=State(),
            speed_counter=Speed(),
            position=(4, 4),
            direction=1,
        )

        def make_env(transitions):
            raw = SimpleNamespace(number_of_agents=1, agents=[agent], rail=Rail(transitions))
            raw.action_required = lambda *args: True
            return FlatlandCIGEnvironment(raw, observation_width=8)

        self.assertEqual(
            make_env((False, True, False, False)).valid_action_mask(0).tolist(),
            [False, False, False, False, True],
        )
        with self.assertRaisesRegex(RuntimeError, "no outgoing rail transition"):
            make_env((False, False, False, False)).valid_action_mask(0)

    def test_external_hard_step_caps_are_enforced(self):
        from envs.flatland_adapter import FlatlandCIGEnvironment
        from envs.external.cyborg import CybORGCIGEnvironment

        flat = FlatlandCIGEnvironment(_FlatlandNeverDone(), observation_width=4, max_steps=2)
        flat.reset(seed=1)
        self.assertFalse(flat.step([0])[2])
        self.assertTrue(flat.step([0])[2])

        cyborg = CybORGCIGEnvironment(_CyborgNeverDone(), observation_width=4, max_steps=2)
        cyborg.reset(seed=1)
        self.assertFalse(cyborg.step([0, 0])[2])
        self.assertTrue(cyborg.step([0, 0])[2])

    def test_cyborg_clone_restore_preserves_adapter_step_budget(self):
        from envs.external.cyborg import CybORGCIGEnvironment

        env = CybORGCIGEnvironment(_CyborgNeverDone(), observation_width=4, max_steps=3)
        env.reset(seed=1)
        env.step([0, 0])
        state = env.clone_state()
        self.assertEqual(env._step_count, 1)
        env.step([0, 0])
        self.assertEqual(env._step_count, 2)
        env.restore_state(state)
        self.assertEqual(env._step_count, 1)
        self.assertFalse(env.step([0, 0])[2])
        self.assertTrue(env.step([0, 0])[2])

    def test_generic_external_oracle_rejects_pseudo_replicated_trials(self):
        from envs.external.rware import RWARECIGEnvironment
        from tests.test_external_adapters import _RWAREDouble

        env = RWARECIGEnvironment(_RWAREDouble(), observation_width=8)
        env.reset(seed=3)
        with self.assertRaisesRegex(ValueError, "pseudo-replication"):
            env.oracle_lag_response(
                ego_id=0, agent_j=1, intervention_action=1, horizon=2, n_trials=2,
                continuation_policy=env.fixed_continuation_policy,
            )

    def test_external_h1_requires_active_response_support(self):
        from scripts.run_external_suite import _h1_smoke

        zero = _h1_smoke(_H1SupportDouble(active_after=None), max_states=4)
        self.assertTrue(zero["interface_ready"])
        self.assertFalse(zero["signal_ready"])
        self.assertGreater(zero["valid_interventions"], 0)
        self.assertEqual(zero["active_interventions"], 0)

        active = _h1_smoke(_H1SupportDouble(active_after=2), max_states=5)
        self.assertGreaterEqual(active["states_tested"], 3)
        self.assertTrue(active["signal_ready"])
        self.assertGreater(active["max_response_mass"], 0.0)

    def test_cityflow_lane_matching_has_no_road_prefix_collision(self):
        from envs.external.cityflow import CityFlowCIGEnvironment

        roadnet = {
            "intersections": [
                {"id": "i1", "virtual": False, "roads": ["road_1"], "trafficLight": {"lightphases": [{}, {}]}},
                {"id": "i10", "virtual": False, "roads": ["road_10"], "trafficLight": {"lightphases": [{}, {}]}},
            ],
            "roads": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "roadnet.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(roadnet, handle)
            env = CityFlowCIGEnvironment(_CityCollisionEngine(), path, max_steps=3, observation_width=8)
            obs = env.reset(seed=1)
            self.assertEqual(float(obs[0][0]), 2.0)
            self.assertEqual(float(obs[0][1]), 3.0)
            _, rewards, _, _ = env.step([0, 0])
            self.assertEqual(rewards[0], -3.0)
            self.assertEqual(rewards[1], -200.0)

    def test_full_explicit_local_has_adapter_and_executes(self):
        import run_experiment as RE

        RE.set_global_seed(0)
        env = RE.make_main_env(
            "behavioral_drift", n_agents=6, max_steps=3, phase_length=2, seed=0
        )
        cfg = RE.smoke_cfg()
        cfg["freeze_policy_learning"] = True
        runner = RE.make_runner("FullExplicitLocal", env, cfg, "cpu")
        self.assertIsNotNone(runner.env_adapter)
        history = runner.run(n_episodes=1, eval_every=1)
        self.assertEqual(len(history["mean_reward"]), 1)
        self.assertTrue(np.isfinite(history["mean_reward"][0]))

    def test_oracle_core_ranks_by_all_action_capacity_not_single_signed_effect(self):
        import run_experiment as RE
        from runners.baseline_runner import OracleCoreRunner

        class Adapter:
            def valid_action_mask(self, agent):
                return np.asarray([True, True, True], dtype=bool)

        class Env:
            n_agents = 3
            causal_adapter = Adapter()
            def __init__(self):
                self.calls = []
            def get_obs_dim(self): return 2
            def get_action_dim(self): return 3
            def clone_state(self): return {"x": 1}
            def restore_state(self, state): return None
            def compute_oracle_capacity_all_egos_from_current_state(self, agent_j, **kwargs):
                self.calls.append(int(agent_j))
                table = {
                    0: np.asarray([0.0, 0.9, 0.2]),
                    1: np.asarray([0.8, 0.0, 0.3]),
                    2: np.asarray([0.1, 0.7, 0.0]),
                }
                return table[int(agent_j)]
            def compute_oracle_influence_all_egos_from_current_state(self, *args, **kwargs):
                raise AssertionError("legacy single-action oracle must not be used")

        env = Env()
        cfg = RE.smoke_cfg(); cfg["seed_core_top_k"] = 1
        runner = OracleCoreRunner(env, cfg, device="cpu")
        runner._refresh_core_if_needed()
        self.assertEqual(env.calls, [0, 1, 2])
        self.assertEqual(runner._cached_core[0], [1])
        self.assertEqual(runner._cached_core[1], [0])
        self.assertEqual(runner._cached_core[2], [1])

    def test_default_config_uses_truthful_hysteresis_names_and_no_dead_h1_thresholds(self):
        import run_experiment as RE

        cfg = RE.default_cfg()
        self.assertIn("belief_tau_enter", cfg)
        self.assertIn("belief_hysteresis_ratio", cfg)
        self.assertNotIn("belief_tau_in", cfg)
        self.assertNotIn("belief_tau_out", cfg)
        self.assertNotIn("h1_max_q_normalized_rmse", cfg)
        self.assertNotIn("h1_max_capacity_normalized_mae", cfg)
        self.assertNotIn("h1_max_direction_normalized_mae", cfg)

    def test_paper_a_missing_optional_latency_artifact_gates_out_only_latency(self):
        from scripts import validate_paper_a as PA

        h1_rows = [{"seed": 1}]
        h2_rows = [
            {
                "seed": 2,
                "model": "Final-CIGAMF",
                "recovery_latency": 1.0,
                "episodes": 20,
                "eval_every": 1,
                "behavioral_false_trigger_rate": 0.01,
                "behavioral_monitoring_window_count": 100,
                "behavioral_false_alarm_window_count": 1,
                "cusum_false_alarm_target": 0.05,
            },
            {
                "seed": 2,
                "model": "NoDetector",
                "recovery_latency": 3.0,
                "episodes": 20,
                "eval_every": 1,
                "behavioral_false_trigger_rate": 0.0,
                "cusum_false_alarm_target": 0.05,
            },
        ]
        with tempfile.TemporaryDirectory() as root, \
             mock.patch.object(PA.CR, "_load_h1_complete_rows", return_value=(h1_rows, {}, {})), \
             mock.patch.object(PA.CR, "_load_h2_complete_rows", return_value=(h2_rows, {})), \
             mock.patch.object(PA.VC, "_h1_status", return_value={"supported": True, "status": "SUPPORTED"}), \
             mock.patch.object(PA.VC, "_h2_status", return_value={"supported": True, "status": "SUPPORTED"}):
            report, code = PA.validate(root, [1], [2], "confirmatory")
        self.assertTrue(report["submitted_claim_set_supported"])
        self.assertEqual(report["overall_status"], "SUPPORTED_LATENCY_GATED_OUT")
        self.assertFalse(report["H3a_latency"]["supported"])
        self.assertFalse(report["H3a_latency"]["oracle_artifact_present"])
        self.assertEqual(code, PA.VC.EXIT_SUPPORTED)

    def test_paper_a_failed_tracking_gates_out_only_optional_trigger(self):
        from scripts import validate_paper_a as PA

        h1_rows = [{"seed": 1}]
        h2_rows = [
            {"seed": 2, "model": "Final-CIGAMF", "recovery_latency": -1.0,
             "episodes": 20, "eval_every": 1, "behavioral_false_trigger_rate": 0.20,
             "cusum_false_alarm_target": 0.05},
            {"seed": 2, "model": "NoDetector", "recovery_latency": 3.0,
             "episodes": 20, "eval_every": 1, "behavioral_false_trigger_rate": 0.0,
             "cusum_false_alarm_target": 0.05},
        ]
        with tempfile.TemporaryDirectory() as root, \
             mock.patch.object(PA.CR, "_load_h1_complete_rows", return_value=(h1_rows, {}, {})), \
             mock.patch.object(PA.CR, "_load_h2_complete_rows", return_value=(h2_rows, {})), \
             mock.patch.object(PA.VC, "_h1_status", return_value={"supported": True, "status": "SUPPORTED"}), \
             mock.patch.object(PA.VC, "_h2_status", return_value={"supported": True, "status": "SUPPORTED"}):
            report, code = PA.validate(root, [1], [2], "confirmatory")
        self.assertTrue(report["submitted_claim_set_supported"])
        self.assertFalse(report["H3b_tracking"]["supported"])
        self.assertTrue(report["overall_status"].startswith("SUPPORTED"))
        self.assertEqual(code, PA.VC.EXIT_SUPPORTED)

    def test_omniarena_latency_onset_uses_shared_protocol_constants(self):
        from envs.omni_arena import OmniArena

        source = inspect.getsource(OmniArena.compute_oracle_lag_response_from_current_state)
        self.assertIn("LATENCY_ONSET_FRACTION", source)
        self.assertIn("LATENCY_ONSET_ABS_FLOOR", source)
        self.assertNotIn("0.10 * float(mass.max())", source)

    def test_collector_ignores_unrequested_stale_legacy_h3_directory(self):
        from scripts import collect_results as CR

        h1_rows = [
            {"seed": 1, "variant": variant}
            for variant in sorted(CR.H1_VARIANTS)
        ]
        h1_manifest = {
            "expected_attempts": [
                {"variant": variant, "seed": 1}
                for variant in sorted(CR.H1_VARIANTS)
            ]
        }
        h2_rows = [{"seed": 2, "model": "Final-CIGAMF"}]
        h2_marker = {"models": ["Final-CIGAMF"], "seeds": [2]}
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "h3"))
            with mock.patch.object(
                CR, "_load_h1_complete_rows",
                return_value=(h1_rows, {}, h1_manifest),
            ), mock.patch.object(
                CR, "_load_h2_complete_rows", return_value=(h2_rows, h2_marker),
            ), mock.patch.object(
                CR, "_load_h3_complete_rows",
                side_effect=AssertionError("stale H3 must not be loaded"),
            ), mock.patch.object(CR, "_validate_seed_matrix"), mock.patch.object(
                CR, "_validate_finite"
            ), mock.patch.object(CR, "_render_table", return_value="table\n"):
                _, text = CR.collect(
                    root, expected_h1_seeds=[1], expected_h2_seeds=[2]
                )
        self.assertNotIn("Legacy H3 diagnostic", text)



if __name__ == "__main__":
    unittest.main()
