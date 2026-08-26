"""Regression tests for the G0--G9 fail-closed scientific gate ladder."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np


class ScientificGateLadderTests(unittest.TestCase):
    def test_forced_only_proxy_sampler_never_falls_back_to_factual_rows(self):
        from models.structural_proxy import LocalCounterfactualProxyEnsemble

        proxy = LocalCounterfactualProxyEnsemble(
            obs_dim=2, action_dim=3, core_dim=1, periph_dim=1,
            belief_dim=1, hidden=8, n_ensemble=1, n_horizons=1,
            device="cpu", forced_only_training=True,
        )
        forced = {
            "was_forced": True, "bootstrap_membership": [True], "tag": "forced",
        }
        factual = {
            "was_forced": False, "bootstrap_membership": [True], "tag": "factual",
        }
        sample = proxy._sample_for_member([factual, forced], 0, 8)
        self.assertTrue(sample)
        self.assertEqual({row["tag"] for row in sample}, {"forced"})
        self.assertEqual(proxy._sample_for_member([factual], 0, 8), [])

    def test_g0_g4_oracle_bank_reuses_all_ego_response_vectors(self):
        from scripts import run_scientific_prechecks as SG

        class Adapter:
            def valid_action_mask(self, source):
                return np.asarray([True, True], dtype=bool)

        class Env:
            n_agents = 3
            def __init__(self):
                self.causal_adapter = Adapter()
                self.calls = 0
            def sample_state_bank(self, **kwargs): return [{"s": 0}]
            def restore_state(self, state): return None
            def set_behaviour_override(self, mode): return None
            def scripted_policy_distribution(self, source):
                return np.asarray([0.8, 0.2], dtype=np.float64)
            def compute_oracle_influence_all_egos_from_current_state(
                self, agent_j, intervention_action, **kwargs
            ):
                self.calls += 1
                # Each source/action returns every ego response.  Capacities are
                # deliberately non-uniform so matched-k random advantage exists.
                base = np.asarray([
                    [0.0, 0.1, 0.4],
                    [0.8, 0.0, 0.2],
                    [0.2, 0.7, 0.0],
                ])[int(agent_j)]
                return base * float(intervention_action)

        env = Env()
        with mock.patch.object(SG.RE, "make_main_env", return_value=env):
            row = SG._oracle_c_d_bank(
                seed=0, n_agents=3, n_states=1, k=1,
                horizon=1, trials=1, discount=0.97,
            )
        # sources * actions, not egos * sources * actions
        self.assertEqual(env.calls, 3 * 2)
        self.assertGreater(row["comparison_count"], 0)
        self.assertTrue(np.isfinite(row["random_advantage_mean"]))

    def test_final_validator_retained_h3_failures_block_full_paper_support(self):
        from scripts import validate_scientific_gates as VG

        with tempfile.TemporaryDirectory() as root:
            prechecks = os.path.join(root, "prechecks.json")
            with open(prechecks, "w", encoding="utf-8") as handle:
                json.dump({
                    "protocol_version": "scientific_prechecks_g0_g4_v3_disagreement_capture",
                    "gates": {
                        name: {"gate": name, "passed": True, "required": True}
                        for name in ("G0", "G1", "G2", "G3", "G4")
                    },
                }, handle)
            # Stub G5/G7/G9 so this test focuses on the gate-ladder scope rule.
            with mock.patch.object(
                VG, "_h1_capacity_gate",
                return_value={"gate": "G5", "passed": True, "required": True},
            ), mock.patch.object(
                VG, "_latency_gate",
                return_value={"gate": "G6", "passed": False, "required": False},
            ), mock.patch.object(
                VG, "_cusum_gate",
                return_value={"gate": "G7", "passed": False, "required": False},
            ), mock.patch.object(
                VG, "_external_gate",
                return_value={"gate": "G8", "passed": False, "required": False},
            ), mock.patch.object(
                VG, "_pareto_gate",
                return_value={"gate": "G9", "passed": False, "required": False},
            ):
                report, code = VG.validate(
                    root, prechecks, None, "confirmatory", min_external_seeds=3
                )
        self.assertEqual(code, VG.EXIT_UNSUPPORTED)
        self.assertTrue(report["core_gates_pass"])
        self.assertEqual(report["overall_status"], "NOT_SUPPORTED")
        self.assertEqual(
            report["claim_scope"]["latency"],
            "RETAINED_BUT_RECOVERY_UNSUPPORTED",
        )
        self.assertEqual(report["claim_scope"]["trigger_tracking"], "DELETE_OR_GATE_OUT")
        self.assertEqual(report["claim_scope"]["generalisation"], "CUSTOM_DOMAIN_ONLY")
        self.assertEqual(
            report["claim_scope"]["system_claim"],
            "SELECTIVE_REPRESENTATION_ARCHITECTURE_ONLY",
        )

    def test_external_g8_requires_paired_full_profile_and_active_support(self):
        from scripts import validate_scientific_gates as VG

        with tempfile.TemporaryDirectory() as root:
            rows = []
            for seed in (1, 2, 3):
                common = {
                    "episodes": 50,
                    "max_steps_requested": 30,
                    "max_steps_effective": 30,
                    "config_fingerprint_sha256": "same-config",
                }
                rows.append({"model": "Final-CIGAMF", "seed": seed, "mean_reward": 2.0, **common})
                rows.append({"model": "PureMeanField", "seed": seed, "mean_reward": 1.0, **common})
            with open(os.path.join(root, "summary_external_training.csv"), "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["model", "seed", "mean_reward", "episodes", "max_steps_requested", "max_steps_effective", "config_fingerprint_sha256"])
                writer.writeheader(); writer.writerows(rows)
            with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "profile": "full", "environment": "rware",
                    "provenance_complete": True, "source_git_clean": True,
                    "external_pin_match": True,
                    "paired_generalization_models_present": True,
                    "seeds": [1, 2, 3], "episodes": 50,
                    "h1_support_by_seed": {
                        str(seed): {"signal_ready": True} for seed in (1, 2, 3)
                    },
                }, handle)
            gate = VG._external_gate(root, min_seeds=3, min_episodes=50)
            self.assertTrue(gate["passed"])
            self.assertGreater(gate["metrics"]["reward_advantage_ci95"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
