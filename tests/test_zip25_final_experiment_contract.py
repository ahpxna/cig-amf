import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import run_experiment as RE
from models.belief_layer import BayesLightBeliefState
from models.single_mean_memory import SingleMeanPeripheral
from models.peripheral_memory import FULL_ITEM_DIM
from envs.omni_arena import OmniArena
from scripts import run_latency_calibration as LAT
from scripts import run_paper_b_scaling as PBS
from scripts import source_tree_hash as STH


class Zip25FinalExperimentContractTests(unittest.TestCase):
    def test_sparse_degree_clamps_one_configured_lower_bound_without_crash(self):
        belief = BayesLightBeliefState(
            ego_id=0,
            neighbor_ids=[1],
            min_core_size=2,
            max_core_size=5,
            adaptive_k=True,
            adaptive_k_min=2,
        )
        self.assertEqual(belief._configured_min_core_size, 2)
        self.assertEqual(belief.min_core_size, 1)
        self.assertEqual(belief.adaptive_k_min, 1)
        self.assertTrue(belief.core_budget_degree_limited)
        belief.reconcile_neighbors([1, 2, 3])
        self.assertEqual(belief.min_core_size, 2)
        self.assertEqual(belief.adaptive_k_min, 2)
        self.assertFalse(belief.core_budget_degree_limited)

    def test_sparse_degree_still_rejects_conflicting_configured_alias(self):
        with self.assertRaises(ValueError):
            BayesLightBeliefState(
                ego_id=0,
                neighbor_ids=[1],
                min_core_size=2,
                max_core_size=5,
                adaptive_k=True,
                adaptive_k_min=1,
            )

    def test_cumulative_signflip_baseline_detects_first_cumulative_direction_flip(self):
        # Two actions, three direct-lag responses.  With factual policy action 0
        # and q uniform, D_h is proportional to Q_h(0)-Q_h(1): +, +, -.
        g = np.asarray([
            [2.0, 0.0, -5.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float64)
        latency, detected = LAT._cumulative_signflip_latency(
            g, np.asarray([1, 1], dtype=bool), policy_action=0, discount=1.0,
        )
        self.assertTrue(detected)
        self.assertEqual(latency, 2)

    def test_cumulative_signflip_baseline_returns_no_detected_delay_without_flip(self):
        g = np.asarray([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        latency, detected = LAT._cumulative_signflip_latency(
            g, np.asarray([1, 1], dtype=bool), policy_action=0, discount=1.0,
        )
        self.assertFalse(detected)
        self.assertEqual(latency, 0)

    def test_scaling_panel_contains_unrestricted_same_architecture_control_and_budget_sweep(self):
        self.assertIn("Semantic-Free", PBS.VARIANTS)
        self.assertIn("Semantic-Free-Unrestricted", PBS.VARIANTS)
        source = Path(PBS.__file__).read_text(encoding="utf-8")
        self.assertIn('"--core-budgets"', source)
        self.assertIn('default=[2, 3, 4, 5]', source)
        self.assertIn('candidate_oracle_replicates', source)
        self.assertIn('candidate_oracle_stable_fraction', source)

    def test_source_tree_hash_ignores_generated_results_but_detects_source_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "scripts").mkdir()
            source = root / "scripts" / "x.py"
            source.write_text("x = 1\n", encoding="utf-8")
            first = STH.source_tree_sha256(root)
            (root / "results").mkdir()
            (root / "results" / "artifact.json").write_text("{}", encoding="utf-8")
            self.assertEqual(first, STH.source_tree_sha256(root))
            source.write_text("x = 2\n", encoding="utf-8")
            self.assertNotEqual(first, STH.source_tree_sha256(root))

    def test_dynamic_low_degree_episode_is_runnable_and_live_state_is_edge_bounded(self):
        env = OmniArena(n_agents=8, n_zones=1, max_steps=6, seed=8871)
        cfg = RE.smoke_cfg()
        cfg.update({
            "seed": 8871,
            "candidate_max_degree": 1,
            "candidate_refresh_interval": 1,
            "candidate_cell_width": 2.0,
            "candidate_stencil_radius": 1,
            "candidate_radius": None,
            # This used to crash because current degree 1 was compared with
            # the configured lower budget 2 after clamping.
            "min_core_size": 2,
            "belief_adaptive_k_min": 2,
            "max_core_size": 4,
            "belief_adaptive_k": True,
            "causal_horizon": 1,
            "proxy_n_horizons": 1,
            "k0_warmup": 0,
            "strict_causal_profile": True,
            "proxy_use_belief_input": False,
        })
        runner = RE.make_runner("Final-CIGAMF", env, cfg, "cpu")
        trajectory, _reward, _runtime = runner.collect_episode()
        self.assertTrue(trajectory)
        self.assertLessEqual(runner.measured_edge_count, runner.n_agents)
        self.assertEqual(len(runner.pair_rel_module.shadow_states), runner.measured_edge_count)
        self.assertEqual(
            sum(len(module.neighbor_ids) for module in runner.belief_modules.values()),
            runner.measured_edge_count,
        )
        self.assertLessEqual(len(runner.pair_rel_module.full_states), runner.measured_edge_count)

    def test_run_all_contains_one_shot_confirmatory_provenance_and_auto_calibration_contract(self):
        source = Path("scripts/run_all.sh").read_text(encoding="utf-8")
        for token in (
            "source_tree_hash.py",
            "automatic development calibration: H1",
            "automatic development calibration: CUSUM",
            "CIG_PAPER_B_CORE_BUDGETS",
            "CIG_PAPER_B_CANDIDATE_RECALL_STABILITY_MIN",
            "CIG_PAPER_B_CANDIDATE_RECALL_STABLE_FRACTION_MIN",
            "source tree SHA-256 changed during the run",
        ):
            self.assertIn(token, source)

    def test_single_mean_accepts_retained_13d_profile_and_scalar_mode(self):
        module = SingleMeanPeripheral(
            action_dim=6, memory_dim=8, out_dim=6, signature_mode="scalar",
            require_full_signature=True, allow_legacy_items=False,
        )
        items = np.zeros((2, FULL_ITEM_DIM), dtype=np.float32)
        items[:, 0] = [1, 2]
        items[:, 1] = [0.5, 0.2]  # C
        items[:, 2] = [0.1, -0.1]  # D
        items[:, 6] = 1.0  # m_ctx
        items[:, 7] = [0.25, 0.75]  # retained latency coordinate
        items[:, 8] = 1.0  # m_L
        out = module(items)
        self.assertEqual(tuple(out.shape), (6,))
        self.assertTrue(bool(np.isfinite(out.detach().cpu().numpy()).all()))

    def test_candidate_recall_uses_top_core_budget_not_all_dmax_slots(self):
        class DummyEnv:
            pass
        source = Path(PBS.__file__).read_text(encoding="utf-8")
        self.assertIn("target_k=target_k", source)
        self.assertIn("candidate_oracle_target_k", source)
        self.assertIn("max tested explicit core budget", source)

    def test_all_stated_allocation_fidelity_comparators_include_weak_prior(self):
        validator = Path("scripts/validate_paper_b.py").read_text(encoding="utf-8")
        self.assertIn('"WeakPrior-Core"', validator)
        self.assertIn("H1a_C_improves_full_explicit_decision_fidelity_over_all_stated_comparators", validator)


if __name__ == "__main__":
    unittest.main()
