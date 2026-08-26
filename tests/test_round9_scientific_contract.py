import copy
import pickle
import unittest

import numpy as np
import torch

import run_experiment as RE
from envs.omni_arena import OmniArena
from models.peripheral_memory import PeripheralMultiMemory
from scripts.representation_isolation import (
    _masked_policy_kl,
    _reference_target_hash,
)
from scripts.run_paper_b_allocation import (
    _mean_score_tables,
    _oracle_replica_summary,
    _topk_jaccard,
    _decision_probe,
)
from scripts.run_paper_b_periphery import _memory_accounting
from scripts.run_paper_b_scaling import (
    _decision_probe_with_latency,
    _runtime_memory_probe,
)


class Round9ScientificContractTests(unittest.TestCase):
    def _runner(self, seed=91):
        env = OmniArena(n_agents=5, n_zones=1, max_steps=6, seed=seed)
        cfg = RE.smoke_cfg()
        cfg.update({
            "seed": seed,
            "causal_horizon": 1,
            "proxy_n_horizons": 1,
            "k0_warmup": 0,
            "min_core_size": 1,
            "belief_adaptive_k_min": 1,
            "max_core_size": 2,
            "periph_require_full_signature": True,
            "periph_allow_legacy_items": False,
        })
        return RE.make_runner("Final-CIGAMF", env, cfg, "cpu")

    def test_deterministic_action_selection_bypasses_forcing_and_rng(self):
        runner = self._runner()
        obs = runner.env_adapter.reset()
        np_state = np.random.get_state()
        forcer_before = copy.deepcopy(runner.forcer.state_dict())
        actions, cache = runner._select_actions_population(obs, apply_forcing=False)
        self.assertFalse(np.asarray(cache["forced_mask"], dtype=bool).any())
        self.assertEqual(cache["action_execution_records"], tuple())
        expected = np.argmax(np.asarray(cache["policy_probs"]), axis=-1)
        self.assertEqual([actions[i] for i in range(runner.n_agents)], expected.tolist())
        self.assertEqual(pickle.dumps(runner.forcer.state_dict()), pickle.dumps(forcer_before))
        after = np.random.get_state()
        self.assertEqual(np_state[0], after[0])
        np.testing.assert_array_equal(np_state[1], after[1])
        self.assertEqual(np_state[2:], after[2:])

    def test_allocation_decision_probe_does_not_advance_forcer(self):
        runner = self._runner(seed=92)
        runner.env_adapter.reset()
        before = copy.deepcopy(runner.forcer.state_dict())
        probe = _decision_probe(runner, n_states=1, seed=1337)
        self.assertEqual(pickle.dumps(runner.forcer.state_dict()), pickle.dumps(before))
        self.assertEqual(probe["actions"].shape[0], 1)

    def test_periphery_diagnostics_can_be_suspended_for_inference(self):
        module = PeripheralMultiMemory(action_dim=3, num_slots=4)
        items = np.zeros((2, 13), dtype=np.float32)
        items[:, 0] = [0, 1]
        items[:, 1] = 0.3
        items[:, 2] = [0.2, -0.2]
        module.set_diagnostics_enabled(False)
        module.forward_full(items)
        self.assertEqual(int(module.slot_diag_updates.item()), 0)
        self.assertEqual(float(module.g_uncertain_usage_ema.item()), 0.0)
        module.set_diagnostics_enabled(True)
        module.forward_full(items)
        self.assertEqual(int(module.slot_diag_updates.item()), 1)

    def test_masked_distillation_kl_ignores_invalid_extreme_logits(self):
        student = torch.tensor([[0.0, 1.0, 1e30]], requires_grad=True)
        teacher = torch.tensor([[1.0, 0.0, -1e30]])
        mask = np.asarray([[True, True, False]])
        loss = _masked_policy_kl(student, teacher, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(float(student.grad[0, 2]), 0.0)

    def test_reference_target_hash_is_deterministic_and_value_sensitive(self):
        targets = [[{
            "logits": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "values": np.asarray([3.0], dtype=np.float32),
            "valid_action_masks": np.asarray([[True, True]]),
        }]]
        first = _reference_target_hash(targets)
        second = _reference_target_hash(copy.deepcopy(targets))
        self.assertEqual(first, second)
        changed = copy.deepcopy(targets)
        changed[0][0]["values"][0] += 1.0
        self.assertNotEqual(first, _reference_target_hash(changed))

    def test_oracle_replica_stability_detects_rank_instability(self):
        stable_a = ({0: {1: 3.0, 2: 2.0, 3: 1.0}}, {0: {1: 0.1, 2: -0.2, 3: 0.0}})
        stable_b = ({0: {1: 4.0, 2: 2.5, 3: 0.5}}, {0: {1: 0.2, 2: -0.3, 3: 0.0}})
        summary = _oracle_replica_summary([stable_a, stable_b], core_budget=1)
        self.assertAlmostEqual(summary["min_c_topk_jaccard"], 1.0)
        unstable = ({0: {1: 0.5, 2: 5.0, 3: 1.0}}, {0: {1: 0.2, 2: -0.3, 3: 0.0}})
        summary_bad = _oracle_replica_summary([stable_a, unstable], core_budget=1)
        self.assertLess(summary_bad["min_c_topk_jaccard"], 1.0)

    def test_mean_oracle_score_tables_preserves_pair_keys(self):
        mean = _mean_score_tables([
            {0: {1: 1.0, 2: 3.0}},
            {0: {1: 3.0, 2: 5.0}},
        ])
        self.assertEqual(set(mean), {0})
        self.assertAlmostEqual(mean[0][1], 2.0)
        self.assertAlmostEqual(mean[0][2], 4.0)
        self.assertAlmostEqual(
            _topk_jaccard({0: {1: 3.0, 2: 1.0}}, {0: {1: 2.0, 2: 0.0}}, 0, 1), 1.0
        )

    def test_analytic_memory_is_not_mislabeled_runtime_peak(self):
        row = _memory_accounting(self._runner(seed=93))
        self.assertEqual(row["representation_memory_is_runtime_peak"], 0)
        self.assertEqual(
            row["representation_memory_accounting_protocol"],
            "analytic_trainable_parameters_plus_persistent_representation_state",
        )
        self.assertEqual(
            row["representation_memory_bytes"],
            row["trainable_parameter_bytes"] + row["persistent_representation_state_bytes"],
        )

    def test_scaling_latency_uses_deterministic_no_forcing_protocol(self):
        runner = self._runner(seed=94)
        runner.env_adapter.reset()
        before = copy.deepcopy(runner.forcer.state_dict())
        _probe, latency = _decision_probe_with_latency(runner, n_states=1, seed=331)
        self.assertEqual(
            latency["inference_latency_protocol"],
            "deterministic_policy_inference_without_training_cache_sampling_or_epsilon_forcing",
        )
        self.assertEqual(pickle.dumps(runner.forcer.state_dict()), pickle.dumps(before))

    def test_runtime_memory_probe_is_separate_and_nonnegative(self):
        runner = self._runner(seed=95)
        runner.env_adapter.reset()
        result = _runtime_memory_probe(runner, n_states=1, seed=332)
        self.assertGreater(result["runtime_peak_process_rss_bytes"], 0)
        self.assertGreaterEqual(result["runtime_peak_process_rss_delta_bytes"], 0)
        self.assertEqual(
            result["runtime_memory_protocol"],
            "separate_deterministic_probe_sampled_process_rss_and_torch_cuda_peaks",
        )


if __name__ == "__main__":
    unittest.main()
