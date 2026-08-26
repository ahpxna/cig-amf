"""Regression tests for confirmatory hardening found by the final audit."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest

import numpy as np

import run_experiment as RE
from models.belief_layer import BayesLightBeliefState
from scripts import collect_cusum_no_change as CUSUM_COLLECT
from scripts import run_h1_calibration as H1
from scripts import run_h2_selectivity as H2
from scripts.h2_cusum_contract import (
    build_h2_cusum_contract,
    contract_hash,
    validate_calibration_artifact,
)


class AuditHardeningTests(unittest.TestCase):
    def test_seeded_belief_cores_are_immediately_active_pair_states(self):
        cfg = RE.smoke_cfg()
        cfg.update({"seed": 111, "k0_warmup": 0})
        env = RE.make_main_env("S0B0", 6, 2, 1000, 111, False, False)
        runner = RE.make_runner("Final-CIGAMF", env, cfg, "cpu")
        expected = {
            (int(ego), int(neighbor))
            for ego, belief in runner.belief_modules.items()
            for neighbor in belief.get_core_set()
        }
        self.assertTrue(expected)
        self.assertEqual(expected, set(runner.pair_rel_module.active_core_pairs))
        self.assertEqual(expected, set(runner.pair_rel_module.full_states))

    def test_full_explicit_core_is_full_from_episode_zero(self):
        cfg = RE.smoke_cfg()
        cfg.update({
            "seed": 113,
            "k0_warmup": 5,
            "core_selection_mode": "full_explicit",
            "max_core_size": 5,
            "min_core_size": 5,
        })
        env = RE.make_main_env("S0B0", 6, 2, 1000, 113, False, False)
        runner = RE.make_runner("Final-CIGAMF", env, cfg, "cpu")
        self.assertTrue(all(len(b.get_core_set()) == 5 for b in runner.belief_modules.values()))
        self.assertEqual(len(runner.pair_rel_module.active_core_pairs), 6 * 5)
        runner.collect_episode()
        self.assertTrue(all(len(b.get_core_set()) == 5 for b in runner.belief_modules.values()))
        self.assertEqual(len(runner.pair_rel_module.active_core_pairs), 6 * 5)

    def test_random_selector_advances_and_weak_prior_is_exposed(self):
        cfg = RE.smoke_cfg()
        cfg.update({"seed": 112, "k0_warmup": 0})
        env = RE.make_main_env("S0B0", 6, 2, 1000, 112, False, False)
        runner = RE.make_runner("Final-CIGAMF", env, cfg, "cpu")
        ego = 0
        neighbors = list(runner.belief_modules[ego].neighbor_ids)
        direction = np.zeros(len(neighbors), dtype=np.float64)
        association = np.zeros((runner.n_agents, runner.n_agents), dtype=np.float64)
        first = runner._external_selector_scores(
            "random", ego, neighbors, direction, association
        )
        second = runner._external_selector_scores(
            "random", ego, neighbors, direction, association
        )
        self.assertTrue(any(first[j] != second[j] for j in neighbors))
        prior = runner._external_selector_scores(
            "weak_prior", ego, neighbors, direction, association
        )
        expected = {
            int(j): float(runner.env_adapter.weak_prior_score(ego, j))
            for j in neighbors
        }
        self.assertEqual(prior, expected)

    def test_adaptive_budget_uses_paper_ceiling_rule(self):
        belief = BayesLightBeliefState(
            ego_id=0, neighbor_ids=[1, 2, 3, 4, 5], min_core_size=2,
            max_core_size=5, adaptive_k=True, adaptive_k_min=2,
        )
        masses = [0.70, 0.20, 0.10, 0.0, 0.0]
        for j, value in zip(belief.neighbor_ids, masses):
            belief._mu_init[j] = float(value)
        p = np.asarray(masses, dtype=np.float64)
        p = p[p > 0.0]
        entropy_fraction = float(-np.sum(p * np.log(p)) / np.log(5.0))
        expected = int(np.ceil(2 + entropy_fraction * 3))
        self.assertEqual(belief._effective_max_k(), expected)

    def test_collector_and_h2_arm_share_canonical_cusum_contract(self):
        collector_cfg = CUSUM_COLLECT._config(seed=1)
        arm_cfg = RE.default_cfg()
        H2._configure_h2_monitoring_cfg(arm_cfg)
        kwargs = dict(
            n_agents=24, max_steps=30, pretrain_episodes=60,
            evaluation_roles=H2.H2_EVALUATION_EGO_ROLES,
            manipulated_roles=H2.H2_MANIPULATED_NEIGHBOR_ROLES,
        )
        self.assertEqual(
            contract_hash(build_h2_cusum_contract(collector_cfg, **kwargs)),
            contract_hash(build_h2_cusum_contract(arm_cfg, **kwargs)),
        )

    def test_cusum_artifact_rejects_impossible_ranges(self):
        bad = {
            "calibration_protocol": "page_cusum_no_change_v2",
            "source_protocol": "cusum_no_change_residual_collection_v2",
            "no_change_only": True,
            "source": {"path": "/missing", "sha256": "0" * 64},
            "cusum_allowance": -999,
            "cusum_threshold": -123,
            "target_false_alarm_rate": 9.0,
            "observed_false_alarm_rate": -4.0,
            "development_seeds": [1],
            "reference_config_hash": "1" * 64,
            "n_no_change_trajectories": 40,
            "monitoring_horizon": 10,
        }
        with self.assertRaises(ValueError):
            validate_calibration_artifact(bad, verify_source=False)

    def test_h1_artifacts_require_provenance_and_support_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            threshold = os.path.join(directory, "threshold.json")
            with open(threshold, "w", encoding="utf-8") as handle:
                json.dump({
                    "oracle_only": True,
                    "capacity_active_threshold": 0.01,
                    "capacity_prediction_threshold": 0.01,
                    "direction_active_threshold": 0.005,
                    "direction_prediction_threshold": 0.005,
                    "h1_target_policy_mode": "scripted_uniform_mixture",
                    "h1_eval_uniform_mass": 0.1,
                }, handle)
            with self.assertRaises(ValueError):
                H1._load_threshold_calibration(threshold)

            support = os.path.join(directory, "support.json")
            with open(support, "w", encoding="utf-8") as handle:
                json.dump({
                    "oracle_only": True,
                    "benchmark_identifiable": True,
                    "decision": "PASS",
                    "development_seeds": [1],
                    "h1_target_policy_mode": "scripted_uniform_mixture",
                    "h1_eval_uniform_mass": 0.1,
                    "thresholds": {"capacity": 0.01, "direction": 0.005},
                }, handle)
            with self.assertRaises(ValueError):
                H1._load_oracle_support(support)


class ExternalAndStateBankHardeningTests(unittest.TestCase):
    def test_state_bank_reserves_oracle_horizon(self):
        env = RE.make_main_env(
            task_mode="behavioral_drift", n_agents=6, max_steps=12,
            phase_length=1000, seed=919,
        )
        horizon = 8
        bank = env.sample_state_bank(
            n_states=12, burn_in=3, bank_seed=991,
            min_remaining_steps=horizon,
        )
        outer = env.clone_state()
        try:
            for state in bank:
                env.restore_state(state)
                self.assertLessEqual(int(env.t) + horizon, int(env.max_steps))
        finally:
            env.restore_state(outer)


    def test_scaling_variants_share_exact_policy_initialization_per_seed(self):
        import torch
        from scripts import run_paper_b_scaling as scaling

        sparse = scaling._runner("Semantic-Free", 6, 12345, "cpu", 2)
        full = scaling._runner("Full-Explicit", 6, 12345, "cpu", 2)
        sparse_state = sparse.policy_value.state_dict()
        full_state = full.policy_value.state_dict()
        self.assertEqual(set(sparse_state), set(full_state))
        self.assertTrue(
            all(torch.equal(sparse_state[key], full_state[key]) for key in sparse_state),
            "matched scaling variants must not differ because of construction order",
        )
        self.assertTrue(
            all(len(b.get_core_set()) == 5 for b in full.belief_modules.values())
        )

    def test_flatland_action_mask_fails_closed_on_api_error(self):
        from types import SimpleNamespace
        from envs.flatland_adapter import FlatlandCIGEnvironment

        class BrokenRail:
            def apply_action_independent(self, action, configuration):
                raise TypeError("simulated pinned-API mismatch")

        class BrokenRailEnv:
            number_of_agents = 1
            agents = [SimpleNamespace(
                position=(1, 1), direction=0, state=object(),
                speed_counter=SimpleNamespace(
                    max_speed=1, is_cell_exit=lambda *args: True
                ),
            )]
            rail = BrokenRail()
            @staticmethod
            def action_required(*args):
                return True

        env = FlatlandCIGEnvironment(BrokenRailEnv(), observation_width=8)
        with self.assertRaisesRegex(RuntimeError, "fail-open"):
            env.valid_action_mask(0)


    def test_flatland_route_profile_uses_pinned_tuple_configuration_api(self):
        from types import SimpleNamespace
        from envs.flatland_adapter import FlatlandCIGEnvironment

        class Rail:
            def __init__(self):
                self.calls = []
            def get_transitions(self, configuration):
                self.calls.append(configuration)
                (row, col), heading = configuration
                # one step east from the first cell, then stop
                return (False, col == 1, False, False)
            def get_full_transitions(self, row, col):
                return 1

        rail = Rail()
        env = FlatlandCIGEnvironment.__new__(FlatlandCIGEnvironment)
        env.rail_env = SimpleNamespace(rail=rail)
        agent = SimpleNamespace(position=(1, 1), direction=1)
        cells = env._reachable_rail_cells(agent, depth_limit=2)
        self.assertIn(((1, 1), 1), rail.calls)
        self.assertIn((1, 2), cells)


    def test_flatland_ready_to_depart_uses_initial_configuration(self):
        from types import SimpleNamespace
        from envs.flatland_adapter import FlatlandCIGEnvironment

        class OffMapState:
            def is_off_map_state(self):
                return True

        class Speed:
            max_speed = 1
            def is_cell_exit(self, speed):
                return True

        class Rail:
            def __init__(self):
                self.configurations = []
            def apply_action_independent(self, action, configuration):
                self.configurations.append(configuration)
                return (configuration, True) if action in (1, 2, 3) else None

        agent = SimpleNamespace(
            current_configuration=None,
            initial_configuration=((3, 4), 1),
            target_configuration=None,
            state=OffMapState(),
            speed_counter=Speed(),
            position=None, direction=None,
        )
        rail = Rail()
        class RailEnv:
            number_of_agents = 1
            agents = [agent]
            @staticmethod
            def action_required(*args):
                return True
        raw = RailEnv()
        raw.rail = rail
        env = FlatlandCIGEnvironment(raw, observation_width=8)
        mask = env.valid_action_mask(0)
        self.assertTrue(mask[1] and mask[2] and mask[3])
        self.assertFalse(mask[0] or mask[4])
        self.assertTrue(rail.configurations)
        self.assertTrue(all(cfg == ((3, 4), 1) for cfg in rail.configurations))

    def test_flatland_action_mask_uses_safe_stop_when_successors_are_temporarily_unavailable(self):
        from types import SimpleNamespace
        from envs.flatland_adapter import FlatlandCIGEnvironment

        class State:
            def is_off_map_state(self):
                return False

        class Speed:
            max_speed = 1
            def is_cell_exit(self, speed):
                return True

        class Rail:
            def apply_action_independent(self, action, configuration):
                return None
            def get_transitions(self, configuration):
                return (False, True, False, False)

        agent = SimpleNamespace(
            current_configuration=((4, 4), 1),
            initial_configuration=((4, 4), 1),
            target_configuration=None,
            state=State(),
            speed_counter=Speed(),
            position=(4, 4), direction=1,
        )
        class RailEnv:
            number_of_agents = 1
            agents = [agent]
            rail = Rail()
            @staticmethod
            def action_required(*args):
                return True

        env = FlatlandCIGEnvironment(RailEnv(), observation_width=8)
        mask = env.valid_action_mask(0)
        self.assertEqual(mask.tolist(), [False, False, False, False, True])

    def test_flatland_action_mask_still_fails_closed_on_invalid_configuration(self):
        from types import SimpleNamespace
        from envs.flatland_adapter import FlatlandCIGEnvironment

        class State:
            def is_off_map_state(self):
                return False

        class Speed:
            max_speed = 1
            def is_cell_exit(self, speed):
                return True

        class Rail:
            def apply_action_independent(self, action, configuration):
                return None
            def get_transitions(self, configuration):
                return (False, False, False, False)

        agent = SimpleNamespace(
            current_configuration=((4, 4), 1),
            initial_configuration=((4, 4), 1),
            target_configuration=None,
            state=State(),
            speed_counter=Speed(),
            position=(4, 4), direction=1,
        )
        class RailEnv:
            number_of_agents = 1
            agents = [agent]
            rail = Rail()
            @staticmethod
            def action_required(*args):
                return True

        env = FlatlandCIGEnvironment(RailEnv(), observation_width=8)
        with self.assertRaisesRegex(RuntimeError, "no outgoing rail transition"):
            env.valid_action_mask(0)

    def test_external_training_capability_requires_clone_restore(self):
        from envs.external_contract import BenchmarkCapabilities
        cap = BenchmarkCapabilities(
            "bad-training", True, True, False, True, False, False, False
        )
        self.assertFalse(cap.supports("training"))


if __name__ == "__main__":
    unittest.main()
