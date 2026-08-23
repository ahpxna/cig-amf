import unittest

import numpy as np
import torch

import run_experiment as RE
from envs.omni_arena import OmniArena
from models.crossfit_aipw import CrossFittedConditionalAIPW
from models.drift_probe import DriftDetector
from models.influence_signature import InfluenceSignatureTracker
from models.intervention import EpsilonForcedActionController
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
from scripts.run_h2_selectivity import FACTORIAL_CELLS, _fixed_estimand_panel
from scripts.run_paper_b_allocation import _oracle_capacity_for_state
from training.replay_builder import MultiEgoReplayBuilder


class RevisedScientificContractTests(unittest.TestCase):
    def test_default_runner_shares_one_causal_horizon_contract(self):
        cfg = RE.default_cfg()
        env = OmniArena(n_agents=6, n_zones=1, max_steps=4, seed=19)
        runner = RE.make_runner("Final-CIGAMF", env, cfg, "cpu")
        expected = int(cfg["causal_horizon"])
        self.assertEqual(runner.proxy.n_horizons, expected)
        self.assertEqual(runner.drift.n_horizons, expected)
        self.assertEqual(runner.replay_builder.horizon, expected)

    def test_drift_residual_is_discounted_full_horizon_return(self):
        detector = DriftDetector(
            obs_dim=1, action_dim=2, n_horizons=3, discount=0.5,
            warmup_batches=1, batch_size=1,
        )
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[0.0, 2.0, 7.0]])
        residual = detector._discounted_return_residual(pred, target)
        # |(1 + 1 + .75) - (0 + 1 + 1.75)| = 0.
        self.assertAlmostEqual(float(residual.item()), 0.0)
        with self.assertRaises(ValueError):
            detector._target_lag_rewards({"target_lag_rewards": [1.0, 2.0]})

    def test_forcer_checkpoint_restores_exact_next_random_draw(self):
        controller = EpsilonForcedActionController(
            n_agents=4, action_dim=3, eps=0.7,
            rng=np.random.RandomState(123),
        )
        probabilities = np.full((4, 3), 1.0 / 3.0, dtype=np.float32)
        controller.apply([0, 0, 0, 0], probabilities)
        checkpoint = controller.state_dict()
        expected_actions = [0, 0, 0, 0]
        expected_mask, expected_probs = controller.apply(
            expected_actions, probabilities
        )
        restored = EpsilonForcedActionController(
            n_agents=4, action_dim=3, eps=0.1,
            rng=np.random.RandomState(999),
        )
        restored.load_state_dict(checkpoint)
        actual_actions = [0, 0, 0, 0]
        actual_mask, actual_probs = restored.apply(actual_actions, probabilities)
        np.testing.assert_array_equal(actual_mask, expected_mask)
        np.testing.assert_allclose(actual_probs, expected_probs)
        self.assertEqual(actual_actions, expected_actions)

    def test_controlled_factorial_event_occurs_on_scheduled_reset(self):
        env = OmniArena(
            n_agents=6, n_zones=1, max_steps=4, phase_length=1000,
            seed=23, enable_structural_shift=False,
        )
        lanes_before = dict(env.active_lane)
        env.schedule_factorial_intervention(True, True, "selfish")
        env.reset()
        actions = [env.scripted_policy(agent) for agent in range(env.n_agents)]
        _, _, _, info = env.step(actions)
        self.assertEqual(info["controlled_structural_shift"], 1)
        self.assertEqual(info["controlled_behavioral_shift"], 1)
        self.assertNotEqual(env.active_lane, lanes_before)
        self.assertEqual(env._behaviour_override, "selfish")

    def test_h2_fixed_estimand_panel_uses_requested_horizon(self):
        env = OmniArena(
            n_agents=6, n_zones=1, max_steps=12, phase_length=1000,
            causal_horizon=2, seed=29,
        )
        panel = _fixed_estimand_panel(
            env, False, False, seed=29, n_states=1, horizon=2,
            discount=0.9, n_trials=1,
        )
        self.assertTrue(panel["applicable"])
        self.assertEqual(panel["horizon"], 2)
        self.assertEqual(panel["continuation_regime"],
                         "cooperative_fixed_after_intervention")
        self.assertTrue(panel["oracle_core_by_ego"])

    def test_paper_b_oracle_selector_uses_all_action_capacity(self):
        env = OmniArena(
            n_agents=6, n_zones=1, max_steps=8, phase_length=1000,
            causal_horizon=1, seed=31,
        )
        state = env.clone_state()
        capacity = _oracle_capacity_for_state(
            env, state, horizon=1, discount=0.9, trials=1, seed=31,
        )
        self.assertEqual(set(capacity), set(range(env.n_agents)))
        self.assertTrue(all(
            len(row) == env.n_agents - 1 for row in capacity.values()
        ))
        self.assertTrue(all(
            value >= 0.0 for row in capacity.values() for value in row.values()
        ))
        # C* is a response range and is not the signed static Phi table.
        self.assertTrue(any(
            abs(float(capacity[ego][source]) - abs(float(
                env.gt_influence_by_ego.get(ego, {}).get(source, 0.0)
            ))) > 1e-8
            for ego in capacity for source in capacity[ego]
        ))

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
