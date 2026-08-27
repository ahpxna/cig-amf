"""Regression tests for the G0--G9 fail-closed scientific gate ladder."""
from __future__ import annotations

import csv
import hashlib
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
                    "protocol_version": "scientific_prechecks_g0_g4_v4_allocation_fidelity",
                    "gates": {
                        **{
                            name: {"gate": name, "passed": True, "required": True}
                            for name in ("G1", "G2", "G3", "G4")
                        },
                        "G0": {
                            "gate": "G0", "passed": True, "required": True,
                            "metrics": {
                                "allocation_value_reference": "Full-Explicit",
                                "allocation_value_protocol": (
                                    "common_frozen_checkpoint_state_bank_policy_context_oracle_allocation"
                                ),
                                "allocation_value_oracle_horizon": 1,
                                "oracle_C_minus_random_logit_fidelity_error_ci95": [0.1, 0.2],
                                "oracle_C_minus_random_value_fidelity_error_ci95": [0.1, 0.2],
                                "oracle_C_minus_random_action_agreement_ci95": [0.1, 0.2],
                            },
                        },
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

    def _write_external_gate_fixture(self, root, *, tamper_summary=False):
        from utils.paper_contracts import (
            EXTERNAL_EVAL_SEED_OFFSET,
            EXTERNAL_GENERALIZATION_PROTOCOL_VERSION,
        )

        seeds = (1, 2, 3)
        eval_episodes = 20
        summary_rows = []
        eval_rows = []
        training_rows = []
        for seed in seeds:
            common = {
                "episodes": 50,
                "eval_episodes": eval_episodes,
                "max_steps_requested": 30,
                "max_steps_effective": 30,
                "config_fingerprint_sha256": "same-config",
            }
            summary_rows.append({
                "model": "Final-CIGAMF", "seed": seed,
                "mean_reward": -10.0, "eval_mean_reward": 2.0, **common,
            })
            summary_rows.append({
                "model": "PureMeanField", "seed": seed,
                "mean_reward": 100.0, "eval_mean_reward": 1.0, **common,
            })
            for model, reward in (("Final-CIGAMF", 2.0), ("PureMeanField", 1.0)):
                for index in range(eval_episodes):
                    eval_rows.append({
                        "model": model,
                        "train_seed": seed,
                        "eval_index": index,
                        "eval_seed": EXTERNAL_EVAL_SEED_OFFSET + seed * 10_000 + index,
                        "eval_reward": reward,
                        "episode_steps": 30,
                        "max_steps_requested": 30,
                        "max_steps_effective": 30,
                        "forcing_count": 0,
                        "rollout_seconds": 0.01,
                        "config_fingerprint_sha256": "same-config",
                    })
            for model in ("Final-CIGAMF", "PureMeanField"):
                for episode in range(1, 51):
                    training_rows.append({
                        "model": model, "seed": seed, "episode": episode,
                        "training_reward": -5.0,
                    })

        def write_csv(name, rows):
            path = os.path.join(root, name)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            with open(path, "rb") as handle:
                return len(rows), hashlib.sha256(handle.read()).hexdigest()

        summary_count, summary_hash = write_csv(
            "summary_external_training.csv", summary_rows
        )
        eval_count, eval_hash = write_csv(
            "external_frozen_evaluation.csv", eval_rows
        )
        train_count, train_hash = write_csv(
            "external_training_episodes.csv", training_rows
        )
        manifest = {
            "protocol_version": EXTERNAL_GENERALIZATION_PROTOCOL_VERSION,
            "profile": "full", "environment": "rware",
            "evaluation_mode": "fresh_seed_frozen_policy_no_learning_no_forcing",
            "evaluation_seed_offset": EXTERNAL_EVAL_SEED_OFFSET,
            "provenance_complete": True, "source_git_clean": True,
            "external_pin_match": True,
            "paired_generalization_models_present": True,
            "not_an_external_allocation_selector_claim": True,
            "seeds": list(seeds), "episodes": 50,
            "eval_episodes": eval_episodes,
            "summary_row_count": summary_count,
            "summary_sha256": summary_hash,
            "evaluation_row_count": eval_count,
            "evaluation_sha256": eval_hash,
            "training_episode_row_count": train_count,
            "training_episode_sha256": train_hash,
            "h1_support_by_seed": {
                str(seed): {"signal_ready": True} for seed in seeds
            },
        }
        with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        if tamper_summary:
            # Mutate the reward artifact *after* the manifest binding is frozen.
            # G8 must reject this before using any reward value.
            summary_rows[0]["eval_mean_reward"] = 999.0
            path = os.path.join(root, "summary_external_training.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
                writer.writeheader(); writer.writerows(summary_rows)

    def test_external_g8_uses_bound_frozen_policy_evaluation(self):
        from scripts import validate_scientific_gates as VG

        with tempfile.TemporaryDirectory() as root:
            self._write_external_gate_fixture(root)
            gate = VG._external_gate(root, min_seeds=3, min_episodes=50)
            self.assertTrue(gate["passed"])
            self.assertFalse(gate["metrics"]["training_return_used_for_gate"])
            self.assertGreater(
                gate["metrics"]["frozen_eval_reward_advantage_ci95"][0], 0.0
            )

    def test_external_g8_rejects_mutated_summary_after_manifest_binding(self):
        from scripts import validate_scientific_gates as VG

        with tempfile.TemporaryDirectory() as root:
            self._write_external_gate_fixture(root, tamper_summary=True)
            with self.assertRaisesRegex(ValueError, "SHA-256 binding mismatch"):
                VG._external_gate(root, min_seeds=3, min_episodes=50)


if __name__ == "__main__":
    unittest.main()
