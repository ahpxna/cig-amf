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


if __name__ == "__main__":
    unittest.main()
