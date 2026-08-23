import unittest

import numpy as np
import torch

from models.crossfit_aipw import CrossFittedConditionalAIPW
from models.influence_signature import InfluenceSignatureTracker
from models.peripheral_memory import (
    FULL_ITEM_DIM,
    ITEM_DIRECTION,
    ITEM_SIGMA_DIRECTION,
    PeripheralMultiMemory,
    ROLE_ANOMALOUS,
    ROLE_BENEFICIAL,
    ROLE_HARMFUL,
    ROLE_NEUTRAL,
)
from models.structural_proxy import LocalCounterfactualProxyEnsemble
from scripts.run_h2_selectivity import FACTORIAL_CELLS
from training.replay_builder import MultiEgoReplayBuilder


class RevisedScientificContractTests(unittest.TestCase):
    def test_direct_lag_targets_derive_cumulative_h_returns(self):
        trajectory = [
            {"rewards": [1.0]},
            {"rewards": [2.0]},
            {"rewards": [3.0]},
        ]
        builder = MultiEgoReplayBuilder(discount=0.5, horizon=3)
        lag = builder.build_lag_rewards(trajectory, 1)[0][0]
        cumulative = builder.build_h_step_returns_multi(trajectory, 1)[0][0]
        np.testing.assert_allclose(lag, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(cumulative, [1.0, 2.0, 2.75])

    def test_online_direction_is_plugin_and_row_aipw_is_separate(self):
        proxy = LocalCounterfactualProxyEnsemble(
            obs_dim=1,
            action_dim=2,
            core_dim=1,
            periph_dim=1,
            belief_dim=1,
            pair_feat_dim=0,
            n_ensemble=1,
            n_horizons=1,
            use_doubly_robust=True,
            effect_mode="signed_policy_contrast",
            use_vmap_ensemble=False,
        )
        proxy._predict_all_actions = lambda **_: torch.tensor(
            [[[[0.0], [1.0]]]], dtype=torch.float32
        )
        out = proxy.score_batch_full(
            obs_i_batch=[[0.0]],
            action_i_batch=[0],
            observed_action_j_batch=[1],
            z_core_excl_j_batch=[[0.0]],
            m_periph_excl_j_batch=[[0.0]],
            belief_summary_batch=[[0.0]],
            policy_probs_j_batch=[[0.2, 0.8]],
            observed_returns_batch=[3.0],
            behaviour_probs_obs_batch=[0.5],
        )
        np.testing.assert_allclose(out["d_mu"], out["d_plugin_mu"])
        self.assertEqual(out["g_lag_mu"].shape, (1, 2, 1))
        self.assertEqual(out["c_lag_mu"].shape, (1, 1))
        np.testing.assert_allclose(out["c_lag_mu"], [[1.0]])
        np.testing.assert_allclose(out["latency_center"], [0.0])
        np.testing.assert_array_equal(out["latency_onset"], [0])
        np.testing.assert_array_equal(out["latency_peak"], [0])
        self.assertNotAlmostEqual(
            float(out["d_plugin_mu"][0]), float(out["d_row_aipw_mu"][0])
        )

    def test_semantic_router_contract(self):
        module = PeripheralMultiMemory(action_dim=3, n_free_slots=2)
        items = torch.zeros(4, FULL_ITEM_DIM)
        items[:, 1] = 1.0
        items[:, ITEM_DIRECTION] = torch.tensor([0.0, 1.0, -1.0, 0.0])
        items[:, ITEM_SIGMA_DIRECTION] = torch.tensor([0.0, 0.0, 0.0, 2.0])
        roles = module._semantic_slot_probs(items).argmax(dim=1).tolist()
        self.assertEqual(
            roles,
            [ROLE_NEUTRAL, ROLE_BENEFICIAL, ROLE_HARMFUL, ROLE_ANOMALOUS],
        )

    def test_context_validity_requires_multiple_supported_contexts(self):
        tracker = InfluenceSignatureTracker(n_agents=3)
        for _ in range(2):
            tracker.update(0, 1, capacity=1.0, direction=0.0,
                           sigma_capacity=0.1, sigma_direction=0.1,
                           context_key="a")
        self.assertEqual(tracker.get_context_validity(0, 1), 0.0)
        for _ in range(2):
            tracker.update(0, 1, capacity=2.0, direction=0.0,
                           sigma_capacity=0.1, sigma_direction=0.1,
                           context_key="b")
        self.assertEqual(tracker.get_context_validity(0, 1), 1.0)

    def test_h2_has_complete_factorial_cells(self):
        self.assertEqual(
            FACTORIAL_CELLS,
            {
                "S0B0": (False, False),
                "S0B1": (False, True),
                "S1B0": (True, False),
                "S1B1": (True, True),
            },
        )

    def test_crossfit_never_trains_nuisance_on_heldout_episode(self):
        samples = []
        for episode in range(4):
            for action in range(2):
                for repeat in range(3):
                    samples.append({
                        "episode_id": episode,
                        "action_i": 0,
                        "observed_action_j": action,
                        "obs_i": np.asarray([episode, repeat], dtype=np.float32),
                        "pair_feat": np.asarray([0.0], dtype=np.float32),
                        "z_core_excl_j": np.asarray([0.0], dtype=np.float32),
                        "m_periph_excl_j": np.asarray([0.0], dtype=np.float32),
                        "policy_probs_j": np.asarray([0.25, 0.75], dtype=np.float32),
                        "behaviour_prob_j": 0.5,
                        "target_return_h": float(action + episode),
                    })
        estimator = CrossFittedConditionalAIPW(2, n_folds=4).fit(samples)
        self.assertTrue(estimator.diagnostics["trajectory_leakage_absent"])
        self.assertEqual(estimator.diagnostics["n_folds"], 4)


if __name__ == "__main__":
    unittest.main()
