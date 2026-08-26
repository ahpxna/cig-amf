"""Regression tests for the C/D redesign contract."""

import unittest

import numpy as np

from models.belief_layer import BayesLightBeliefState
from models.influence_signature import (
    InfluenceSignatureTracker,
    ROLE_BENEFICIAL,
)
from models.structural_proxy import (
    LocalCounterfactualProxyEnsemble,
    build_pair_feat,
)
from envs.causal_adapter import OmniArenaAdapter
from runners.baseline_runner import SharedAblationBase, _adapt_executed_policy
from runners.final_runner import FinalCIGAMFRunner


class CDDecompositionTests(unittest.TestCase):
    @staticmethod
    def _context_runner(cls):
        class Env:
            positions = [[0, 0], [8, 0], [0, 6], [8, 6]]
            agent_zone = [0, 1, 0, 2]
            grid_size = 10
            n_zones = 3
            last_actions = [0, 1, 2, 1]

        runner = cls.__new__(cls)
        runner.env = Env()
        runner.n_agents = 4
        runner.action_dim = 3
        runner.core_dim = 11
        runner.periph_dim = 13
        runner.candidate_neighbors_by_ego = {
            ego: [j for j in range(runner.n_agents) if j != ego]
            for ego in range(runner.n_agents)
        }
        return runner

    def test_raw_proxy_context_is_partition_independent_and_leave_one_out(self):
        for cls in (FinalCIGAMFRunner, SharedAblationBase):
            runner = self._context_runner(cls)
            z_without_1, m_without_1 = runner._raw_proxy_context_excluding(0, 1)
            z_without_2, m_without_2 = runner._raw_proxy_context_excluding(0, 2)
            self.assertEqual(z_without_1.shape, (11,))
            self.assertEqual(m_without_1.shape, (13,))
            self.assertTrue(np.all(np.isfinite(z_without_1)))
            self.assertTrue(np.all(np.isfinite(m_without_1)))
            self.assertFalse(np.array_equal(z_without_1, z_without_2))
            self.assertFalse(np.array_equal(m_without_1, m_without_2))

    def test_behavioral_execution_adapter_changes_sampled_policy_only_in_h2_mode(self):
        class Env:
            mode = "behavioral_drift"

            @staticmethod
            def scripted_policy_distribution(_agent):
                return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        learned = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        final = FinalCIGAMFRunner.__new__(FinalCIGAMFRunner)
        final.env = Env()
        final.n_agents = 2
        final.action_dim = 3
        final.cfg = {
            "behavioral_adapter_lambda": 1.0,
            "behavioral_adapter_only_in_behavioral_drift": True,
        }
        executed, _scripted, diagnostics = final._execution_policy_adapter(learned)
        np.testing.assert_allclose(executed[:, 1], 1.0)
        self.assertEqual(diagnostics["behavioral_adapter_active"], 1)
        self.assertGreater(diagnostics["behavioral_adapter_tv"], 0.9)

        baseline_executed, baseline_diagnostics = _adapt_executed_policy(
            final.env, final.cfg, learned
        )
        np.testing.assert_allclose(baseline_executed, executed)
        self.assertEqual(baseline_diagnostics["behavioral_adapter_active"], 1)

    def test_behavioral_adapter_can_hold_evaluation_egos_fixed(self):
        class Env:
            mode = "behavioral_drift"
            agent_role = ["collector", "blocker", "collector"]

            @staticmethod
            def scripted_policy_distribution(_agent):
                return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        learned = np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        final = FinalCIGAMFRunner.__new__(FinalCIGAMFRunner)
        final.env = Env()
        final.n_agents = 3
        final.action_dim = 3
        final.cfg = {
            "behavioral_adapter_lambda": 1.0,
            "behavioral_adapter_only_in_behavioral_drift": True,
            "behavioral_adapter_target_roles": ["blocker"],
        }
        executed, _scripted, diagnostics = final._execution_policy_adapter(learned)
        np.testing.assert_allclose(executed[[0, 2]], learned[[0, 2]])
        np.testing.assert_allclose(executed[1], [0.0, 1.0, 0.0])
        self.assertEqual(diagnostics["behavioral_adapter_target_count"], 1)
        self.assertEqual(diagnostics["behavioral_adapter_non_target_count"], 2)
        self.assertEqual(diagnostics["behavioral_adapter_non_target_tv"], 0.0)

    def test_adaptive_core_uses_one_minimum_and_zero_evidence_stays_small(self):
        belief = BayesLightBeliefState(
            ego_id=0,
            neighbor_ids=[1, 2, 3],
            min_core_size=1,
            max_core_size=3,
            adaptive_k=True,
            adaptive_k_min=1,
        )
        self.assertEqual(belief._effective_max_k(), 1)
        with self.assertRaisesRegex(ValueError, "single lower bound"):
            BayesLightBeliefState(
                ego_id=0,
                neighbor_ids=[1, 2],
                min_core_size=1,
                max_core_size=2,
                adaptive_k=True,
                adaptive_k_min=2,
            )

    def test_direction_allocation_keeps_capacity_budget_but_changes_ranking(self):
        belief = BayesLightBeliefState(
            ego_id=0,
            neighbor_ids=[1, 2, 3],
            min_core_size=2,
            max_core_size=2,
        )
        belief.core_set = {1, 2}
        promoted, demoted = belief.select_core_from_external_scores(
            {1: 0.1, 2: 0.2, 3: 0.9}, target_size=2
        )
        self.assertEqual(belief.get_core_set(), {2, 3})
        self.assertEqual(promoted, {3})
        self.assertEqual(demoted, {1})

    def test_pair_feature_uses_public_target_role_without_oracle_label(self):
        class Env:
            positions = [[0, 0], [2, 1]]
            agent_zone = [0, 0]
            grid_size = 10
            n_zones = 1
            agent_role = ["collector", "blocker"]

        adapter = OmniArenaAdapter(Env())
        feat = build_pair_feat(adapter, 0, 1)
        self.assertEqual(feat.shape, (adapter.pair_feature_dim,))
        self.assertEqual(float(np.sum(feat[5:])), 1.0)
        self.assertEqual(float(feat[8]), 1.0)  # blocker

    def test_tracker_keeps_capacity_direction_and_uncertainties_separate(self):
        tracker = InfluenceSignatureTracker(n_agents=3, window=8)
        tracker.update(
            ego_id=0,
            neighbor_id=1,
            capacity=2.0,
            direction=0.4,
            sigma_capacity=0.1,
            sigma_direction=0.3,
            context_key="lane",
        )
        tracker.update(
            ego_id=0,
            neighbor_id=1,
            capacity=4.0,
            direction=-0.2,
            sigma_capacity=0.5,
            sigma_direction=0.7,
            context_key="gate",
        )
        signature = tracker.get_signature(0, 1)
        np.testing.assert_allclose(signature[:4], [3.0, 0.1, 0.3, 0.5])
        self.assertGreater(signature[4], 0.0)

    def test_semantic_role_uses_direction_not_capacity(self):
        tracker = InfluenceSignatureTracker(n_agents=3, tau_role=0.05, sigma_hi=1.0)
        tracker.update(
            ego_id=0,
            neighbor_id=1,
            capacity=9.0,
            direction=0.4,
            sigma_capacity=3.0,
            sigma_direction=0.1,
        )
        self.assertEqual(tracker.get_role(0, 1), ROLE_BENEFICIAL)

    def test_structural_belief_does_not_promote_negative_legacy_scalar(self):
        belief = BayesLightBeliefState(
            ego_id=0,
            neighbor_ids=[1, 2],
            min_core_size=0,
            max_core_size=2,
            tau=0.1,
            kappa=0.0,
            alpha_decay=0.0,
        )
        belief.update_batch({1: (-3.0, 0.01), 2: (0.2, 0.01)})
        self.assertNotIn(1, belief.get_core_set())
        self.assertIn(2, belief.get_core_set())

    def test_proxy_exposes_distinct_c_and_d_fields(self):
        proxy = LocalCounterfactualProxyEnsemble(
            obs_dim=2,
            action_dim=3,
            core_dim=2,
            periph_dim=2,
            belief_dim=1,
            n_ensemble=2,
            hidden=16,
            n_horizons=1,
            effect_mode="signed_policy_contrast",
            use_doubly_robust=False,
            use_vmap_ensemble=False,
            device="cpu",
        )
        out = proxy.score_batch_full(
            obs_i_batch=np.zeros((2, 2), dtype=np.float32),
            action_i_batch=[0, 1],
            observed_action_j_batch=[0, 1],
            z_core_excl_j_batch=np.zeros((2, 2), dtype=np.float32),
            m_periph_excl_j_batch=np.zeros((2, 2), dtype=np.float32),
            belief_summary_batch=np.zeros((2, 1), dtype=np.float32),
            policy_probs_j_batch=np.full((2, 3), 1.0 / 3.0, dtype=np.float32),
        )
        for key in ("c_mu", "c_sigma", "d_mu", "d_sigma", "q_mu", "q_sigma"):
            self.assertIn(key, out)
        self.assertEqual(out["q_mu"].shape, (2, 3))
        self.assertTrue(np.all(out["c_mu"] >= 0.0))


if __name__ == "__main__":
    unittest.main()
