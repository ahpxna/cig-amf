import unittest
import contextlib
import io
import tempfile
from unittest import mock

import numpy as np
import torch

import run_experiment as RE
from envs.omni_arena import OmniArena
from envs.causal_adapter import OmniArenaAdapter
from models.crossfit_aipw import CrossFittedConditionalAIPW
from models.drift_probe import DriftDetector
from models.influence_signature import InfluenceSignatureTracker
from models.intervention import EpsilonForcedActionController
from models.peripheral_memory import (
    FULL_ITEM_DIM,
    ITEM_DIRECTION,
    ITEM_SIGMA_DIRECTION,
    PeripheralMultiMemory,
    ROLE_UNCERTAIN,
    ROLE_BENEFICIAL,
    ROLE_HARMFUL,
    ROLE_NEUTRAL,
)
from models.structural_proxy import LocalCounterfactualProxyEnsemble
from scripts.run_h2_selectivity import (
    FACTORIAL_CELLS,
    H2_MANIPULATED_NEIGHBOR_ROLES,
    _capture_frozen_learning_checkpoint,
    _cached_estimand_panels,
    _fixed_estimand_panel,
    _restore_frozen_learning_checkpoint,
)
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

    def test_final_runner_does_not_require_geometry_on_env_surface(self):
        inner = OmniArena(n_agents=6, n_zones=1, max_steps=2, seed=17)

        class AdapterOnlyEnv:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self.causal_adapter = OmniArenaAdapter(wrapped)

            def __getattr__(self, name):
                if name in {
                    "positions", "agent_zone", "agent_role",
                    "grid_size", "n_zones",
                }:
                    raise AttributeError(name)
                return getattr(self._wrapped, name)

        env = AdapterOnlyEnv(inner)
        cfg = RE.smoke_cfg()
        cfg.update({
            "causal_horizon": 2,
            "proxy_n_horizons": 2,
            "min_core_size": 1,
            "belief_adaptive_k_min": 1,
            "max_core_size": 3,
        })
        runner = RE.make_runner("Final-CIGAMF", env, cfg, "cpu")
        trajectory, _, _ = runner.collect_episode()
        self.assertTrue(trajectory)
        self.assertIn("proxy_context_blocks", trajectory[0])

    def test_shared_ablation_uses_same_literal_context_contract(self):
        env = OmniArena(n_agents=6, n_zones=1, max_steps=2, seed=18)
        cfg = RE.smoke_cfg()
        cfg.update({
            "causal_horizon": 2,
            "proxy_n_horizons": 2,
            "min_core_size": 1,
            "belief_adaptive_k_min": 1,
            "max_core_size": 3,
        })
        runner = RE.make_runner("NoMultiMemory", env, cfg, "cpu")
        trajectory, _, _ = runner.collect_episode()
        pushed = runner.replay_builder.push_trajectory_to_proxy(
            trajectory, runner.proxy, runner.env
        )
        self.assertGreater(pushed, 0)
        sample = runner.proxy.buffer[0]
        self.assertEqual(
            sample["context_items"].shape[1],
            runner.env_adapter.context_item_dim,
        )

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

    def test_action_masks_apply_to_forcing_and_structural_capacity(self):
        controller = EpsilonForcedActionController(
            n_agents=1, action_dim=3, eps=1.0,
            rng=np.random.RandomState(7),
        )
        valid = np.asarray([[True, False, True]])
        for _ in range(25):
            actions = [0]
            _, behaviour = controller.apply(
                actions,
                np.asarray([[0.2, 0.7, 0.1]], dtype=np.float32),
                valid_action_masks=valid,
            )
            self.assertIn(actions[0], (0, 2))
            self.assertEqual(float(behaviour[0, 1]), 0.0)

        proxy = LocalCounterfactualProxyEnsemble(
            obs_dim=1, action_dim=3, core_dim=1, periph_dim=1,
            belief_dim=1, pair_feat_dim=0, n_ensemble=1,
            n_horizons=1, use_vmap_ensemble=False,
        )
        # The invalid middle action has an extreme prediction. C must still be
        # max(Q_0,Q_2)-min(Q_0,Q_2)=2, not 100.
        proxy._predict_all_actions = lambda **_: torch.tensor(
            [[[[0.0], [100.0], [2.0]]]], dtype=torch.float32
        )
        out = proxy.score_batch_full(
            obs_i_batch=[[0.0]], action_i_batch=[0],
            observed_action_j_batch=[0], z_core_excl_j_batch=[[0.0]],
            m_periph_excl_j_batch=[[0.0]], belief_summary_batch=[[0.0]],
            valid_action_mask_batch=valid,
        )
        np.testing.assert_allclose(out["c_mu"], [2.0])
        np.testing.assert_allclose(out["c_lag_mu"], [[2.0]])

    def test_proxy_uses_member_owned_literal_deepsets(self):
        proxy = LocalCounterfactualProxyEnsemble(
            obs_dim=1, action_dim=2, core_dim=1, periph_dim=1,
            belief_dim=1, pair_feat_dim=0, context_item_dim=1,
            n_ensemble=2, hidden=4, n_horizons=1,
            use_vmap_ensemble=False,
        )
        first, second = proxy.models
        self.assertIsNot(first.context_item_encoder, second.context_item_encoder)
        self.assertNotEqual(
            next(first.context_item_encoder.parameters()).data_ptr(),
            next(second.context_item_encoder.parameters()).data_ptr(),
        )

        # These sets have identical sum and sum-of-squares, so the former
        # moment-summary encoder could not distinguish them. A nonlinear phi
        # before summation can.
        with torch.no_grad():
            for layer in first.context_item_encoder:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.zero_()
                    layer.bias.zero_()
            first.context_item_encoder[0].weight[0, 0] = 1.0
            first.context_item_encoder[2].weight[0, 0] = 1.0
        set_a = torch.tensor([[[-1.0], [0.0], [1.0]]])
        set_b = torch.tensor([[[1.0 / np.sqrt(3)],
                               [1.0 / np.sqrt(3)],
                               [-2.0 / np.sqrt(3)]]], dtype=torch.float32)
        self.assertAlmostEqual(float(set_a.sum()), float(set_b.sum()), places=6)
        self.assertAlmostEqual(
            float(torch.square(set_a).sum()),
            float(torch.square(set_b).sum()), places=6,
        )
        emb_a = first.context_item_encoder(set_a).sum(dim=-2)
        emb_b = first.context_item_encoder(set_b).sum(dim=-2)
        self.assertFalse(torch.allclose(emb_a, emb_b))

    def test_proxy_diagnostic_residual_is_discounted_h_return(self):
        proxy = LocalCounterfactualProxyEnsemble(
            obs_dim=1, action_dim=2, core_dim=1, periph_dim=1,
            belief_dim=1, pair_feat_dim=0, n_ensemble=1,
            n_horizons=3, discount=0.5, use_vmap_ensemble=False,
        )
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[0.0, 2.0, 7.0]])
        # Discounted totals are both 2.75 although the last-lag error is 4.
        residual = proxy._discounted_return_residual(pred, target)
        self.assertAlmostEqual(float(residual.item()), 0.0)

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
        np.testing.assert_array_equal(out["latency_onset"], [-1])
        np.testing.assert_array_equal(out["latency_peak"], [-1])
        np.testing.assert_array_equal(out["latency_valid"], [0.0])
        self.assertNotAlmostEqual(
            float(out["d_plugin_mu"][0]), float(out["d_row_aipw_mu"][0])
        )

    def test_subthreshold_lag_mass_is_not_misreported_as_zero_lag(self):
        proxy = LocalCounterfactualProxyEnsemble(
            obs_dim=1, action_dim=2, core_dim=1, periph_dim=1,
            belief_dim=1, pair_feat_dim=0, n_ensemble=1, n_horizons=3,
            use_vmap_ensemble=False,
        )
        proxy._predict_all_actions = lambda **_: torch.tensor(
            [[[[0.0, 0.0, 0.0], [1e-8, 1e-8, 1e-8]]]],
            dtype=torch.float32,
        )
        out = proxy.score_batch_full(
            obs_i_batch=[[0.0]], action_i_batch=[0],
            observed_action_j_batch=[1], z_core_excl_j_batch=[[0.0]],
            m_periph_excl_j_batch=[[0.0]], belief_summary_batch=[[0.0]],
            policy_probs_j_batch=[[0.5, 0.5]],
        )
        np.testing.assert_array_equal(out["latency_valid"], [0.0])
        np.testing.assert_array_equal(out["latency_onset_valid"], [0.0])
        np.testing.assert_array_equal(out["latency_onset"], [-1])
        np.testing.assert_array_equal(out["latency_peak"], [-1])

    def test_capacity_is_range_of_the_ensemble_mean_response_surface(self):
        proxy = LocalCounterfactualProxyEnsemble(
            obs_dim=1, action_dim=2, core_dim=1, periph_dim=1,
            belief_dim=1, pair_feat_dim=0, n_ensemble=2, n_horizons=2,
            use_vmap_ensemble=False,
        )
        # Each member has range 1, but they disagree on which action is high.
        # Paper A defines C on Q-bar, whose action range is therefore zero.
        proxy._predict_all_actions = lambda **_: torch.tensor(
            [
                [[[0.0, 0.0], [1.0, 1.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ], dtype=torch.float32,
        )
        out = proxy.score_batch_full(
            obs_i_batch=[[0.0]], action_i_batch=[0],
            observed_action_j_batch=[0], z_core_excl_j_batch=[[0.0]],
            m_periph_excl_j_batch=[[0.0]], belief_summary_batch=[[0.0]],
            policy_probs_j_batch=[[0.5, 0.5]],
        )
        np.testing.assert_allclose(out["c_mu"], [0.0])
        np.testing.assert_allclose(out["c_lag_mu"], [[0.0, 0.0]])
        self.assertGreater(float(out["c_sigma"][0]), 0.0)

    def test_semantic_router_contract(self):
        module = PeripheralMultiMemory(action_dim=3, n_free_slots=2)
        items = torch.zeros(4, FULL_ITEM_DIM)
        items[:, 1] = 1.0
        items[:, ITEM_DIRECTION] = torch.tensor([0.0, 1.0, -1.0, 0.0])
        items[:, ITEM_SIGMA_DIRECTION] = torch.tensor([0.0, 0.0, 0.0, 2.0])
        roles = module._semantic_slot_probs(items).argmax(dim=1).tolist()
        self.assertEqual(
            roles,
            [ROLE_NEUTRAL, ROLE_BENEFICIAL, ROLE_HARMFUL, ROLE_UNCERTAIN],
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

    def test_h2_oracle_panels_are_cached_once_per_seed(self):
        calls = []

        def fake_panel(*args, **kwargs):
            calls.append((kwargs["structural_factor"], kwargs["behavioral_factor"]))
            return {
                "applicable": True,
                "capacity_mean": 1.0,
                "direction_abs_mean": 0.5,
                "capacity_mean_by_ego": {0: {1: 1.0}},
                "oracle_core_by_ego": {0: [1]},
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.run_h2_selectivity.RE.make_main_env",
            return_value=object(),
        ), mock.patch(
            "scripts.run_h2_selectivity._fixed_estimand_panel",
            side_effect=fake_panel,
        ):
            pre, cells = _cached_estimand_panels(4, directory)
            self.assertEqual(len(calls), 4)  # pre/S0B0 is shared, plus 3 cells.
            self.assertEqual(set(cells), set(FACTORIAL_CELLS))
            reloaded_pre, reloaded_cells = _cached_estimand_panels(4, directory)
            self.assertEqual(len(calls), 4)
            self.assertIn(0, reloaded_pre["oracle_core_by_ego"])
            self.assertIn(0, reloaded_cells["S1B1"]["capacity_mean_by_ego"])
            self.assertEqual(pre["capacity_mean"], reloaded_pre["capacity_mean"])

    def test_end_to_end_micro_scientific_contract(self):
        """Exercise cross-module research contracts without mocked internals."""
        cfg = RE.smoke_cfg()
        cfg.update({
            "causal_horizon": 2,
            "proxy_n_horizons": 2,
            "n_ensemble": 2,
            "proxy_train_steps": 1,
            "proxy_batch_size": 16,
            "proxy_holdout_size": 0,
            "bc_train_steps": 1,
            "bc_batch_size": 16,
            "k0_warmup": 1,
            "graph_score_steps": 1,
            "drift_warmup_batches": 1,
            "drift_train_batches": 1,
            "forcer_anneal_to": 0.03,
            "min_core_size": 1,
            "belief_adaptive_k_min": 1,
            "max_core_size": 3,
            "seed": 101,
        })
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            env = RE.make_main_env("S0B0", 6, 2, 1000, 101, False, False)
            runner = RE.make_runner("Final-CIGAMF", env, cfg, "cpu")
            captured = []
            collect = runner.collect_episode

            def collect_and_capture():
                result = collect()
                captured.append(result[0])
                return result

            runner.collect_episode = collect_and_capture
            runner.run(2, 1)

            self.assertTrue(captured)
            trajectory = captured[-1]
            self.assertTrue(all(
                "valid_action_masks" in step
                and "proxy_context_blocks" in step
                and "forced_mask" in step
                for step in trajectory
            ))
            self.assertTrue(all(
                bool(step["valid_action_masks"][agent][step["actions"][agent]])
                for step in trajectory for agent in range(runner.n_agents)
            ))
            self.assertGreater(len(runner.proxy.buffer), 0)
            proxy_loss = runner.proxy.train_step(
                n_steps=1, batch_size=16, holdout_size=0
            )
            self.assertTrue(np.isfinite(proxy_loss))

            step = trajectory[0]
            runner.env_adapter.restore_state(step["env_snapshot_before_step"])
            h_return = runner.replay_builder.build_h_step_returns(
                trajectory, runner.n_agents
            )[0]
            _, _, capacity = runner._score_all_pairs_and_update_beliefs(
                obs_all=step["obs_all"],
                actions=step["actions"],
                observed_returns=h_return,
                behaviour_probs=step["behaviour_probs"],
                policy_probs=step["policy_probs"],
            )
            self.assertTrue(np.all(np.isfinite(capacity)))
            self.assertTrue(np.all(capacity >= 0.0))
            runner.env_adapter.restore_state(
                trajectory[-1]["env_snapshot_after_step"]
            )

            runner.drift.prepare_for_monitoring(
                runner.proxy.buffer, episode=2, reference_batches=20
            )
            witness = runner.drift.step(
                episode=3, buffer=runner.proxy.buffer, n_train_batches=1
            )
            self.assertEqual(witness["phase"], "monitoring")
            self.assertTrue(np.isfinite(witness["z"]))
            checkpoint = _capture_frozen_learning_checkpoint(runner)

            observed_cells = {}
            for mode, (structural, behavioral) in FACTORIAL_CELLS.items():
                branch_env = RE.make_main_env(
                    mode, 6, 2, 1000, 101, False, False
                )
                branch = RE.make_runner(
                    "Final-CIGAMF", branch_env, cfg, "cpu"
                )
                _restore_frozen_learning_checkpoint(branch, checkpoint)
                branch_env.schedule_factorial_intervention(
                    structural, behavioral, "selfish"
                )
                branch.cfg.update({
                    "behavioral_adapter_only_in_behavioral_drift": False,
                    "behavioral_adapter_target_roles": list(
                        H2_MANIPULATED_NEIGHBOR_ROLES
                    ),
                    "behavioral_adapter_lambda": 1.0 if behavioral else 0.0,
                    "freeze_policy_learning": True,
                    "freeze_representation_state": True,
                })
                branch.run(1, 1)
                event = branch.episode_events[-1]
                panel = _fixed_estimand_panel(
                    branch_env, structural, behavioral, seed=101,
                    n_states=1, horizon=2, discount=cfg["discount"],
                    n_trials=1,
                )
                self.assertTrue(panel["applicable"])
                observed_cells[mode] = (
                    int(event["structural_shift"]),
                    int(event["behavioral_shift"]),
                )

        self.assertEqual(observed_cells, {
            "S0B0": (0, 0),
            "S0B1": (0, 1),
            "S1B0": (1, 0),
            "S1B1": (1, 1),
        })

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
