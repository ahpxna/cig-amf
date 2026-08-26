import copy
import pickle
import unittest

import numpy as np
import torch

import run_experiment as RE
from envs.causal_adapter import OmniArenaAdapter, build_dynamic_candidate_map
from envs.omni_arena import OmniArena
from models.belief_layer import BayesLightBeliefState
from models.core_behavior import PairRelationalModule
from models.structural_proxy import LocalCounterfactualProxyEnsemble


class DynamicCandidateContractTests(unittest.TestCase):
    def _env(self, n=6, seed=301):
        return OmniArena(n_agents=max(5, n), n_zones=1, max_steps=8, seed=seed)

    def _runner(self, seed=302, **overrides):
        env = self._env(n=8, seed=seed)
        cfg = RE.smoke_cfg()
        cfg.update({
            "seed": seed,
            "candidate_max_degree": 2,
            "candidate_refresh_interval": 1,
            "candidate_cell_width": 2.0,
            "candidate_stencil_radius": 1,
            "candidate_radius": None,
            "causal_horizon": 1,
            "proxy_n_horizons": 1,
            "k0_warmup": 0,
            "min_core_size": 0,
            "belief_adaptive_k_min": 0,
            "max_core_size": 2,
            "strict_causal_profile": True,
            "proxy_context_item_dim": None,
            "proxy_pair_feat_dim": None,
            "proxy_use_belief_input": False,
        })
        cfg.update(overrides)
        return RE.make_runner("Final-CIGAMF", env, cfg, "cpu")

    def test_linked_cell_population_provider_is_bounded_and_hashed(self):
        env = self._env(n=8)
        adapter = OmniArenaAdapter(env)
        env.reset()
        mapping, telemetry = build_dynamic_candidate_map(
            adapter, 2, cell_width=2.0, stencil_radius=1
        )
        self.assertLessEqual(sum(map(len, mapping.values())), 8 * 2)
        self.assertTrue(telemetry["candidate_construction_linear_candidate"])
        self.assertTrue(telemetry["feature_snapshot_linear_candidate"])
        self.assertEqual(telemetry["provider"], "omni_linked_cell_dynamic")
        self.assertEqual(len(telemetry["candidate_map_hash"]), 64)
        mapping2, telemetry2 = build_dynamic_candidate_map(
            adapter, 2, cell_width=2.0, stencil_radius=1
        )
        self.assertEqual(mapping, mapping2)
        self.assertEqual(telemetry["candidate_map_hash"], telemetry2["candidate_map_hash"])

    def test_dynamic_provider_changes_with_current_observable_geometry(self):
        env = self._env(n=4, seed=303)
        adapter = OmniArenaAdapter(env)
        env.reset()
        # Construct two separated same-cell pairs under a zero-radius stencil.
        env.positions[0] = [0, 0]
        env.positions[1] = [0, 1]
        env.positions[2] = [8, 8]
        env.positions[3] = [8, 9]
        env.positions[4] = [20, 20]
        first, t1 = build_dynamic_candidate_map(
            adapter, 1, cell_width=4.0, stencil_radius=0
        )
        self.assertEqual(first[0], (1,))
        env.positions[1] = [8, 8]
        second, t2 = build_dynamic_candidate_map(
            adapter, 1, cell_width=4.0, stencil_radius=0
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(t1["candidate_map_hash"], t2["candidate_map_hash"])

    def test_pair_state_reconciliation_preserves_retained_zeros_added_evicts_removed(self):
        module = PairRelationalModule(
            n_agents=4, obs_dim=3, action_dim=2, hidden_dim=5,
            shadow_dim=3, rel_feat_dim=2,
            candidate_neighbors_by_ego={0: [1, 2], 1: [], 2: [], 3: []},
        )
        with torch.no_grad():
            module.shadow_states[(0, 1)].fill_(7.0)
        module.reconcile_core_sets({0: [2]}, warm_start=False)
        self.assertIn((0, 2), module.full_states)
        added, removed = module.reconcile_candidate_sets(
            {0: [1, 3], 1: [], 2: [], 3: []}
        )
        self.assertEqual(added, {(0, 3)})
        self.assertEqual(removed, {(0, 2)})
        self.assertTrue(torch.all(module.shadow_states[(0, 1)] == 7.0))
        self.assertTrue(torch.all(module.shadow_states[(0, 3)] == 0.0))
        self.assertNotIn((0, 2), module.shadow_states)
        self.assertNotIn((0, 2), module.full_states)
        self.assertNotIn((0, 2), module.active_core_pairs)

    def test_belief_reconciliation_preserves_retained_and_evicts_removed(self):
        belief = BayesLightBeliefState(
            ego_id=0, neighbor_ids=[1, 2], min_core_size=0, max_core_size=2,
            adaptive_k_min=0,
        )
        belief.update_pair(1, 0.8, 0.2)
        before = belief.get_state_dict()[1]
        belief.set_fixed_core([2])
        added, removed = belief.reconcile_neighbors([1, 3])
        self.assertEqual(added, {3})
        self.assertEqual(removed, {2})
        after = belief.get_state_dict()
        self.assertAlmostEqual(after[1]["capacity_bar"], before["capacity_bar"])
        self.assertEqual(after[3]["n_updates"], 0)
        self.assertAlmostEqual(after[3]["capacity_bar"], 0.0)
        self.assertGreaterEqual(after[3]["sigma_bar"], 1.0 - 1e-9)
        self.assertNotIn(2, after)
        self.assertNotIn(2, belief.get_core_set())

    def test_runner_refresh_bounds_live_shadow_state_by_current_edges(self):
        runner = self._runner(seed=304)
        runner.env_adapter.reset()
        runner._refresh_dynamic_candidates(force=True)
        self.assertEqual(len(runner.pair_rel_module.shadow_states), runner.measured_edge_count)
        self.assertLessEqual(runner.measured_edge_count, runner.n_agents * 2)
        epoch = runner.candidate_epoch
        runner._refresh_dynamic_candidates(force=True)
        self.assertGreater(runner.candidate_epoch, epoch)
        self.assertEqual(len(runner.pair_rel_module.shadow_states), runner.measured_edge_count)

    def test_removed_live_pair_does_not_survive_runner_refresh(self):
        runner = self._runner(seed=305, candidate_cell_width=1.0, candidate_stencil_radius=0)
        runner.env_adapter.reset()
        # Explicit geometry makes (0,1) a current candidate.
        runner.env.positions[0] = [0, 0]
        runner.env.positions[1] = [0, 0]
        for j in range(2, runner.n_agents):
            runner.env.positions[j] = [5 + j, 5 + j]
        runner._refresh_dynamic_candidates(force=True)
        self.assertIn(1, runner.candidate_neighbors_by_ego[0])
        runner.pair_rel_module.reconcile_core_sets({0: [1]}, warm_start=False)
        self.assertIn((0, 1), runner.pair_rel_module.full_states)
        runner.env.positions[1] = [20, 20]
        runner._refresh_dynamic_candidates(force=True)
        self.assertNotIn((0, 1), runner.pair_rel_module.shadow_states)
        self.assertNotIn((0, 1), runner.pair_rel_module.full_states)

    def test_inference_only_path_omits_training_cache_and_forcing(self):
        runner = self._runner(seed=306)
        obs = runner.env_adapter.reset()
        before = pickle.dumps(copy.deepcopy(runner.forcer.state_dict()))
        _actions, cache = runner._select_actions_population(
            obs, apply_forcing=False, collect_training_cache=False
        )
        self.assertIsNone(cache["geom_snapshot"])
        self.assertEqual(cache["proxy_context_blocks"], {})
        self.assertFalse(np.asarray(cache["forced_mask"], dtype=bool).any())
        self.assertEqual(before, pickle.dumps(runner.forcer.state_dict()))

    def test_candidate_epoch_and_map_are_stored_with_factual_cache(self):
        runner = self._runner(seed=307)
        obs = runner.env_adapter.reset()
        _actions, cache = runner._select_actions_population(obs, apply_forcing=False)
        self.assertGreaterEqual(int(cache["candidate_epoch"]), 1)
        self.assertEqual(cache["candidate_map_hash"], runner.candidate_map_hash)
        self.assertEqual(
            {k: tuple(v) for k, v in cache["candidate_neighbors_by_ego"].items()},
            runner.candidate_neighbors_by_ego,
        )

    def test_strict_proxy_contract_rejects_downstream_belief_input(self):
        with self.assertRaises(ValueError):
            self._runner(seed=308, proxy_use_belief_input=True)

    def test_strict_proxy_contract_rejects_missing_raw_context_or_pair_features(self):
        with self.assertRaises(ValueError):
            self._runner(seed=309, proxy_context_item_dim=0)
        with self.assertRaises(ValueError):
            self._runner(seed=310, proxy_pair_feat_dim=0)

    def test_linked_cell_work_bound_is_linear_under_frozen_occupancy(self):
        env = self._env(n=12, seed=311)
        adapter = OmniArenaAdapter(env)
        env.reset()
        _, telemetry = build_dynamic_candidate_map(
            adapter, 3, cell_width=2.0, stencil_radius=1
        )
        occ = int(telemetry["cell_occupancy_max"])
        bound = env.n_agents * (1 + 9 * occ)
        self.assertLessEqual(int(telemetry["provider_work_units"]), bound)

    def test_forced_probe_refresh_updates_candidate_epoch_without_interaction_step(self):
        runner = self._runner(seed=312)
        obs = runner.env_adapter.reset()
        step = runner._interaction_step
        runner._select_actions_population(
            obs, apply_forcing=False, collect_training_cache=False,
            force_candidate_refresh=True,
        )
        first = runner.candidate_epoch
        self.assertEqual(runner._interaction_step, step)
        runner._select_actions_population(
            obs, apply_forcing=False, collect_training_cache=False,
            force_candidate_refresh=True,
        )
        self.assertGreater(runner.candidate_epoch, first)
        self.assertEqual(runner._interaction_step, step)

    def test_dynamic_episode_keeps_historical_shadow_snapshot_for_bc(self):
        runner = self._runner(seed=313, candidate_cell_width=2.0, candidate_stencil_radius=1)
        trajectory, _reward, _runtime = runner.collect_episode()
        self.assertGreater(len(trajectory), 1)
        self.assertTrue(all("candidate_neighbors_by_ego" in step for step in trajectory))
        self.assertTrue(all("s_snapshot_before_latent_update" in step for step in trajectory))
        self.assertEqual(len(runner.pair_rel_module.shadow_states), runner.measured_edge_count)

    def test_paper_a_response_regression_defaults_to_unweighted_factual_loss(self):
        proxy = LocalCounterfactualProxyEnsemble(
            obs_dim=3, action_dim=2, core_dim=2, periph_dim=2, belief_dim=2,
            n_ensemble=2, n_horizons=1, use_doubly_robust=False,
        )
        self.assertFalse(proxy.response_ipw_ablation)
        proxy_ipw = LocalCounterfactualProxyEnsemble(
            obs_dim=3, action_dim=2, core_dim=2, periph_dim=2, belief_dim=2,
            n_ensemble=2, n_horizons=1, use_doubly_robust=False,
            response_ipw_ablation=True,
        )
        self.assertTrue(proxy_ipw.response_ipw_ablation)


if __name__ == "__main__":
    unittest.main()
