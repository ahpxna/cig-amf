"""Focused regression tests for H2 protocol integrity."""
import csv
import hashlib
import json
import os
import tempfile
import unittest
from collections import deque

import numpy as np

from runners.baseline_runner import (
    CorrelationMeanFieldRunner,
    SharedAblationBase,
    _ordered_geometry_snapshot,
)
from runners.final_runner import FinalCIGAMFRunner, NoTwoTimescaleRunner
from scripts.collect_results import (
    H1_VARIANTS,
    H2_PROTOCOL_VERSION,
    H3_VARIANTS,
    ResultValidationError,
    _load_h2_complete_rows,
    collect,
)
from scripts.run_h2_selectivity import (
    _matched_change_interval_mask,
    _recovery_statistics,
)


class _Scheduler:
    def __init__(self, stage=0):
        self.stage = int(stage)
        self.episode = 0
        self.trigger_count = 0

    def should_update_graph(self):
        return False

    def step_episode(self):
        self.episode += 1

    def get_status(self):
        return {"accel_remaining": 0}


class _PairModule:
    def train_bc(self, **kwargs):
        return 0.0


class _Proxy:
    def get_buffer_size(self):
        return 0


def _snapshot():
    return {
        "mean_reward": 1.0,
        "reward_per_agent": 1.0,
        "mean_f1": 0.5,
        "mean_f1_role": 0.25,
        "mean_temporal_var": 0.0,
        "mean_uncertainty": 0.1,
        "mean_core_size": 2.0,
        "mean_core_switches": 0.0,
        "mean_mu": 0.1,
        "max_p": 0.5,
        "runtime": 0.01,
        "throughput_agent_steps_per_sec": 100.0,
        "episode_runtime_total": 0.02,
        "throughput_total_agent_steps_per_sec": 50.0,
    }


def _trajectory():
    return [{
        "info": {"delta_phi_frobenius_structural": 0.0},
        "actions": [0],
        "rewards": [1.0],
    }]


class H2ProtocolTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path, rows):
        keys = sorted({key for row in rows for key in row})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def test_geometry_snapshot_orders_dictionary_values_by_agent_id(self):
        class Env:
            positions = {1: [8, 9], 0: [2, 3]}
            agent_zone = {1: 4, 0: 1}
            grid_size = 12
            n_zones = 5

        snapshot = _ordered_geometry_snapshot(Env(), 2)
        self.assertEqual(snapshot["positions"], [[2, 3], [8, 9]])
        self.assertEqual(snapshot["agent_zone"], [1, 4])

    def test_shared_runner_preserves_global_episode_clock_across_chunks(self):
        runner = SharedAblationBase.__new__(SharedAblationBase)
        runner.episodes_completed = 0
        runner.episode_events = []
        runner.scheduler = _Scheduler(stage=1)
        runner.use_belief = False
        runner.use_two_timescale = False
        runner.n_agents = 1
        runner.name = "ProtocolDouble"
        runner.cfg = {"bc_train_steps": 0, "bc_batch_size": 1}
        runner.pair_rel_module = _PairModule()
        runner.proxy = _Proxy()
        class Env:
            phase = "cooperative"

            def _behaviour_mode(self):
                return self.phase

        runner.env = Env()
        collection = {"count": 0}

        def collect_episode():
            collection["count"] += 1
            if collection["count"] == 3:
                runner.env.phase = "delayed"
            return _trajectory(), np.asarray([1.0]), 0.01

        runner.collect_episode = collect_episode
        runner.push_trajectory_to_proxy_buffer = lambda trajectory: 0
        runner.update_policy = lambda trajectory: 0.0
        graph_calls = {"count": 0}

        def update_graph(_trajectory_arg):
            graph_calls["count"] += 1
            triggered = int(graph_calls["count"] in {1, 3})
            runner.scheduler.trigger_count += triggered
            return {
                "proxy_loss": 0.0,
                "proxy_train_residual": 0.0,
                "proxy_holdout_residual": 0.0,
                "triggered": triggered,
                "promoted": 0,
                "demoted": 0,
            }

        runner.use_belief = True
        runner.update_graph_modules = update_graph
        runner.evaluate_episode_snapshot = lambda *args, **kwargs: _snapshot()
        runner._reset_switch_counters_if_available = lambda: None
        keys = (
            "episodes mean_reward reward_per_agent mean_f1 mean_temporal_var "
            "mean_uncertainty mean_core_size mean_core_switches mean_mu max_p "
            "runtime throughput_agent_steps_per_sec episode_runtime_total "
            "throughput_total_agent_steps_per_sec proxy_train_residual "
            "proxy_holdout_residual scheduler_residual_ewma scheduler_cusum_score "
            "scheduler_accel_remaining proxy_loss bc_loss policy_loss triggered "
            "trigger_count stage proxy_buffer_size pushed_proxy_samples promoted demoted"
        ).split()
        runner.history = {key: [] for key in keys}

        runner.run(n_episodes=2, eval_every=2)
        runner.run(n_episodes=2, eval_every=2)

        self.assertEqual(runner.history["episodes"], [2, 4])
        self.assertEqual(runner.scheduler.episode, 4)
        self.assertEqual(
            [event["episode"] for event in runner.episode_events],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            [event["triggered"] for event in runner.episode_events],
            [1, 0, 1, 0],
        )
        self.assertEqual(runner.scheduler.trigger_count, 2)
        self.assertEqual(
            [event["behavioral_shift"] for event in runner.episode_events],
            [0, 0, 1, 0],
        )

    def test_final_runner_preserves_global_episode_clock_across_chunks(self):
        runner = FinalCIGAMFRunner.__new__(FinalCIGAMFRunner)
        runner.episodes_completed = 0
        runner.episode_events = []
        runner.scheduler = _Scheduler(stage=0)
        runner.n_agents = 1
        runner.cfg = {"bc_train_steps": 0, "bc_batch_size": 1}
        runner.pair_rel_module = _PairModule()
        runner.proxy = _Proxy()
        runner.heads = object()
        runner.heads_optim = object()
        runner.belief_modules = {}
        runner.collect_episode = lambda: (_trajectory(), np.asarray([1.0]), 0.01)
        runner.push_trajectory_to_proxy_buffer = lambda trajectory: 0
        runner.update_policy = lambda trajectory: {
            "loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "grad_norm_preclip": 0.0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
        }
        runner.evaluate_episode_snapshot = lambda *args, **kwargs: _snapshot()
        runner._reset_switch_counters_if_available = lambda: None
        keys = (
            "episodes mean_reward reward_per_agent mean_f1 mean_f1_role "
            "mean_temporal_var mean_uncertainty mean_core_size mean_core_switches "
            "mean_mu max_p runtime throughput_agent_steps_per_sec "
            "episode_runtime_total throughput_total_agent_steps_per_sec "
            "proxy_train_residual proxy_holdout_residual scheduler_residual_ewma "
            "scheduler_cusum_score scheduler_accel_remaining proxy_loss bc_loss "
            "policy_loss actor_loss critic_loss entropy grad_norm_preclip adv_mean "
            "adv_std triggered trigger_count stage proxy_buffer_size "
            "pushed_proxy_samples promoted demoted"
        ).split()
        runner.history = {key: [] for key in keys}

        runner.run(n_episodes=2, eval_every=2)
        runner.run(n_episodes=2, eval_every=2)

        self.assertEqual(runner.history["episodes"], [2, 4])
        self.assertEqual(runner.scheduler.episode, 4)
        self.assertEqual(
            [event["episode"] for event in runner.episode_events],
            [1, 2, 3, 4],
        )

    def test_correlation_baseline_recovers_supported_association(self):
        runner = CorrelationMeanFieldRunner.__new__(CorrelationMeanFieldRunner)
        runner.n_agents = 3
        runner.action_dim = 2
        runner.association_window_steps = 120
        runner.association_min_steps = 20
        runner.association_min_action_support = 5
        runner._association_actions = deque(maxlen=runner.association_window_steps)
        runner._association_rewards = deque(maxlen=runner.association_window_steps)
        runner._association_matrix = np.zeros((3, 3), dtype=np.float64)

        trajectory = []
        for step in range(80):
            source_action = step % 2
            trajectory.append({
                "actions": [0, source_action, (step // 3) % 2],
                "rewards": [float(source_action), 0.0, 0.0],
            })
        runner._observe_episode(trajectory)

        matrix = runner.get_influence_matrix()
        self.assertGreater(matrix[0, 1], 0.99)
        self.assertTrue(np.allclose(np.diag(matrix), 0.0))
        self.assertTrue(np.all((matrix >= 0.0) & (matrix <= 1.0)))

    def test_no_two_timescale_is_a_scheduler_only_final_subclass(self):
        self.assertTrue(issubclass(NoTwoTimescaleRunner, FinalCIGAMFRunner))
        runner = NoTwoTimescaleRunner.__new__(NoTwoTimescaleRunner)
        self.assertTrue(runner._should_update_graph_this_episode())
        self.assertEqual(
            runner.ablation_contract,
            "scheduler_only_graph_update_every_episode",
        )

    def test_matched_windows_start_with_each_straddling_interval(self):
        finite_rows = [
            (episode - 10, episode, float(episode))
            for episode in range(10, 101, 10)
        ]
        structural_mask, structural = _matched_change_interval_mask(
            finite_rows, [41, 81], 2
        )
        behavioral_mask, behavioral = _matched_change_interval_mask(
            finite_rows, [40, 80, 120], 2
        )
        self.assertEqual(
            [(finite_rows[i][0], finite_rows[i][1]) for i in np.flatnonzero(structural_mask)],
            [(40, 50), (50, 60), (80, 90), (90, 100)],
        )
        self.assertEqual(
            [(finite_rows[i][0], finite_rows[i][1]) for i in np.flatnonzero(behavioral_mask)],
            [(30, 40), (40, 50), (70, 80), (80, 90)],
        )
        self.assertTrue(all(window["complete"] for window in structural))
        self.assertFalse(behavioral[-1]["complete"])

    def test_recovery_aggregates_every_shift_and_matched_trigger(self):
        records = [
            {"episode": 10, "f1": 0.8},
            {"episode": 20, "f1": 0.8},
            {"episode": 30, "f1": 0.4},
            {"episode": 40, "f1": 0.8},
            {"episode": 50, "f1": 0.4},
            {"episode": 60, "f1": 0.8},
        ]
        result = _recovery_statistics(
            records,
            shift_episodes=[30, 50],
            trigger_episodes=[31, 52],
            causal_horizon_steps=0,
            max_steps=30,
            eval_every=10,
        )
        self.assertEqual(len(result["recovery_by_shift"]), 2)
        self.assertEqual(result["n_recovered_shifts"], 2)
        self.assertEqual(result["n_shift_with_trigger"], 2)
        self.assertEqual(result["recovery_latency_intervals"], 1.0)

    def test_recovery_delay_uses_step_units_and_never_goes_negative(self):
        records = [
            {"episode": 10, "f1": 0.8},
            {"episode": 20, "f1": 0.8},
            {"episode": 30, "f1": 0.8},
            {"episode": 40, "f1": 0.8},
        ]
        result = _recovery_statistics(
            records,
            shift_episodes=[40],
            trigger_episodes=[40],
            causal_horizon_steps=8,
            max_steps=30,
            eval_every=10,
        )
        self.assertAlmostEqual(result["trigger_delay_intervals"], 8 / 300)
        self.assertEqual(result["recovery_latency_raw_intervals"], 0.0)
        self.assertEqual(result["recovery_latency_intervals"], 0.0)

    def test_collector_rejects_stale_summary_when_latest_attempt_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "summary_h2.csv"), "w", encoding="utf-8") as f:
                f.write("run_id,model,seed\nold,Final-CIGAMF,0\n")
            with open(os.path.join(directory, "latest_attempt.json"), "w", encoding="utf-8") as f:
                json.dump({"run_id": "new", "status": "failed"}, f)
            with self.assertRaises(ResultValidationError):
                _load_h2_complete_rows(directory)

    def test_collector_accepts_checksum_bound_complete_h2_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = os.path.join(directory, "summary.csv")
            with open(summary, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "run_id", "protocol_version", "model", "seed",
                    "claim_evaluable", "n_complete_structural_windows",
                    "n_complete_behavioral_windows", "runner_class",
                    "ablation_contract", "policy_learning_frozen",
                    "pretrain_episodes", "behavioral_adapter_target_count",
                    "behavioral_adapter_non_target_tv",
                ])
                writer.writeheader()
                writer.writerow({
                    "run_id": "run-1",
                    "protocol_version": H2_PROTOCOL_VERSION,
                    "model": "Final-CIGAMF",
                    "seed": 0,
                    "claim_evaluable": 1,
                    "n_complete_structural_windows": 1,
                    "n_complete_behavioral_windows": 1,
                    "runner_class": "FinalCIGAMFRunner",
                    "ablation_contract": "",
                    "policy_learning_frozen": 1,
                    "pretrain_episodes": 1,
                    "behavioral_adapter_target_count": 1,
                    "behavioral_adapter_non_target_tv": 0.0,
                })
            with open(summary, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            marker = {
                "run_id": "run-1",
                "status": "complete",
                "protocol_version": H2_PROTOCOL_VERSION,
                "required_comparator_present": True,
                "required_claim_models_present": True,
                "models": ["Final-CIGAMF"],
                "seeds": [0],
                "expected_attempts": [
                    {"model": "Final-CIGAMF", "mode": mode, "seed": 0}
                    for mode in ("behavioral_drift", "structural_shift")
                ],
                "completed_attempts": [
                    {"model": "Final-CIGAMF", "mode": mode, "seed": 0}
                    for mode in ("behavioral_drift", "structural_shift")
                ],
                "failed_attempt": None,
                "summary_path": summary,
                "summary_sha256": checksum,
                "expected_summary_rows": 1,
                "summary_rows": 1,
            }
            with open(os.path.join(directory, "latest_attempt.json"), "w", encoding="utf-8") as f:
                json.dump(marker, f)
            rows, loaded = _load_h2_complete_rows(directory)
            self.assertEqual(rows[0]["run_id"], "run-1")
            self.assertEqual(loaded["status"], "complete")

    def test_collector_accepts_only_complete_cross_hypothesis_matrix(self):
        # Unit tests must not depend on a repository-level results directory.
        with tempfile.TemporaryDirectory() as root:
            h1_dir = os.path.join(root, "h1")
            h2_dir = os.path.join(root, "h2")
            h3_dir = os.path.join(root, "h3")
            os.makedirs(h1_dir)
            os.makedirs(h2_dir)
            os.makedirs(h3_dir)

            h1_run = os.path.join(h1_dir, "runs", "h1-run")
            os.makedirs(h1_run)
            h1_attempts = [
                {"variant": variant, "seed": 0}
                for variant in sorted(H1_VARIANTS)
            ]
            h1_rows = [
                {
                    **attempt,
                    "run_id": "h1-run",
                    "protocol_version": "h1_qcd_crossfit_v3",
                    "config_fingerprint": "abc123",
                    "attempt_complete": True,
                    "q_spearman_mean": 0.1,
                    "capacity_rank_correlation_mean": 0.1,
                    "oracle_core_f1_mean": 0.8,
                    "direction_spearman_mean": 0.1,
                    "direction_sign_agreement_mean": 0.8,
                }
                for attempt in h1_attempts
            ]
            h1_summary = os.path.join(h1_run, "summary_h1.csv")
            self._write_csv(h1_summary, h1_rows)
            h1_manifest = {
                "run_id": "h1-run",
                "protocol_version": "h1_qcd_crossfit_v3",
                "status": "complete",
                "expected_attempts": h1_attempts,
                "completed_attempts": h1_attempts,
                "failed_attempts": [],
                "summary_path": h1_summary,
                "row_count": len(h1_rows),
            }
            h1_manifest_path = os.path.join(h1_run, "manifest.json")
            with open(h1_manifest_path, "w", encoding="utf-8") as f:
                json.dump(h1_manifest, f)
            with open(os.path.join(h1_dir, "latest_complete_run.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "run_id": "h1-run",
                    "protocol_version": "h1_qcd_crossfit_v3",
                    "summary_path": h1_summary,
                    "manifest_path": h1_manifest_path,
                }, f)

            h2_models = [
                "Final-CIGAMF", "CorrelationMeanField", "NoTwoTimescale"
            ]
            h2_rows = [
                {
                    "run_id": "h2-run",
                    "protocol_version": H2_PROTOCOL_VERSION,
                    "model": model,
                    "seed": 0,
                    "claim_evaluable": 1,
                    "delta_behav": 0.1,
                    "delta_struct": 0.2,
                    "delta_background_structural_run": 0.05,
                    "delta_background_behavioral_run": 0.05,
                    "SR_C": 2.0,
                    "recovery_latency": 1.0,
                    "n_shift_events": 1,
                    "n_shift_with_trigger": 1,
                    "final_f1_struct": 0.5,
                    "n_complete_structural_windows": 1,
                    "n_complete_behavioral_windows": 1,
                    "runner_class": (
                        "NoTwoTimescaleRunner"
                        if model == "NoTwoTimescale"
                        else (
                            "CorrelationMeanFieldRunner"
                            if model == "CorrelationMeanField"
                            else "FinalCIGAMFRunner"
                        )
                    ),
                    "ablation_contract": (
                        "scheduler_only_graph_update_every_episode"
                        if model == "NoTwoTimescale"
                        else ""
                    ),
                    "policy_learning_frozen": 1,
                    "pretrain_episodes": 1,
                    "behavioral_adapter_target_count": 1,
                    "behavioral_adapter_non_target_tv": 0.0,
                }
                for model in h2_models
            ]
            h2_summary = os.path.join(h2_dir, "summary_h2.csv")
            self._write_csv(h2_summary, h2_rows)
            with open(h2_summary, "rb") as f:
                h2_hash = hashlib.sha256(f.read()).hexdigest()
            with open(os.path.join(h2_dir, "latest_attempt.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "run_id": "h2-run",
                    "status": "complete",
                    "protocol_version": H2_PROTOCOL_VERSION,
                    "required_comparator_present": True,
                    "required_claim_models_present": True,
                    "expected_attempts": [
                        {"model": model, "mode": mode, "seed": 0}
                        for model in h2_models
                        for mode in ("behavioral_drift", "structural_shift")
                    ],
                    "completed_attempts": [
                        {"model": model, "mode": mode, "seed": 0}
                        for model in h2_models
                        for mode in ("behavioral_drift", "structural_shift")
                    ],
                    "failed_attempt": None,
                    "summary_path": h2_summary,
                    "summary_sha256": h2_hash,
                    "expected_summary_rows": len(h2_rows),
                    "summary_rows": len(h2_rows),
                    "models": h2_models,
                    "seeds": [0],
                }, f)

            h3_attempts = [
                {"variant": variant, "seed": 0}
                for variant in sorted(H3_VARIANTS)
            ]
            h3_rows = []
            for attempt in h3_attempts:
                row = {
                    **attempt,
                    "mean_core_size": 2.0,
                    "frac_k_at_kmax": 0.2,
                    "mean_reward": -1.0,
                    "mean_f1": 0.5,
                    "throughput_total": 10.0,
                }
                if attempt["variant"] != "NoMultiMemory-SingleMean":
                    row.update({
                        "hard_usage_entropy_ratio": 0.8,
                        "assignment_mutual_info_ratio": 0.2,
                        "slot_cos_offdiag": 0.5,
                    })
                h3_rows.append(row)
            self._write_csv(os.path.join(h3_dir, "summary_h3.csv"), h3_rows)
            with open(os.path.join(h3_dir, "attempt.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "status": "complete",
                    "failed_run": None,
                    "seeds": [0],
                    "expected_runs": h3_attempts,
                    "completed_runs": h3_attempts,
                    "hypothesis_gate": {},
                }, f)

            output, text = collect(root, expected_seeds=[0])
            self.assertTrue(os.path.exists(output))
            self.assertIn("matched-seed post-warm-up F1", text)


# The collector resolves relative manifest paths against the repository root.
ROOT_FOR_TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    unittest.main()
