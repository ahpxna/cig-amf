import contextlib
import io
import unittest

import numpy as np
import torch

from models.intervention import EpsilonForcedActionController
from models.structural_proxy import LocalCounterfactualProxyEnsemble
from scripts import run_h1_calibration as h1_calibration


def _proxy(action_dim=3, effect_mode="signed_oracle_matched", use_dr=True,
           pair_feat_dim=0, lr=1e-3):
    return LocalCounterfactualProxyEnsemble(
        obs_dim=2,
        action_dim=action_dim,
        core_dim=1,
        periph_dim=1,
        belief_dim=1,
        n_ensemble=1,
        hidden=48,
        lr=lr,
        buffer_size=2048,
        device="cpu",
        n_horizons=1,
        effect_mode=effect_mode,
        use_doubly_robust=use_dr,
        iw_clip=20.0,
        bootstrap_ratio=1.0,
        pair_feat_dim=pair_feat_dim,
        seed=7,
        use_vmap_ensemble=False,
    )


class H1EstimandTests(unittest.TestCase):
    def test_policy_contrast_aipw_uses_target_and_logging_probabilities(self):
        proxy = _proxy(effect_mode="signed_policy_contrast", use_dr=True)
        preds = torch.tensor(
            [[[[1.0], [2.0], [4.0]], [[-1.0], [0.5], [3.0]]]],
            dtype=torch.float32,
        )
        observed_actions = np.array([0, 2])
        observed_returns = np.array([1.6, 2.4], dtype=np.float32)
        propensities = np.array([0.4, 0.2], dtype=np.float32)
        target_policy = np.array(
            [[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32
        )

        result = proxy._compute_effects(
            preds,
            observed_actions,
            policy_probs_j=target_policy,
            observed_returns=observed_returns,
            behaviour_probs_obs=propensities,
            mode="signed_policy_contrast",
        )
        factual = np.array([1.0, 3.0])
        plugin = np.array([
            0.7 * 1.0 + 0.2 * 2.0 + 0.1 * 4.0 - (1.0 + 2.0 + 4.0) / 3.0,
            0.2 * -1.0 + 0.3 * 0.5 + 0.5 * 3.0 - (-1.0 + 0.5 + 3.0) / 3.0,
        ])
        residual = observed_returns - factual
        pi_observed = np.array([0.7, 0.5])
        expected = plugin + (
            (pi_observed - 1.0 / 3.0) / propensities
        ) * residual

        np.testing.assert_allclose(
            result["effect"].detach().numpy()[0], expected, atol=1e-6
        )
        self.assertTrue(result["dr_applied"])
        self.assertEqual(result["dr_applied_rows"], 2)

    def test_hand_calculation_rejects_previous_aipw_coefficients(self):
        proxy = _proxy(effect_mode="signed_policy_contrast", use_dr=True)
        preds = torch.tensor([[[[1.0], [2.0], [4.0]]]], dtype=torch.float32)
        policy = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)
        behaviour = np.array([0.6], dtype=np.float32)
        observed_return = np.array([1.5], dtype=np.float32)

        result = proxy._compute_effects(
            preds,
            np.array([0]),
            policy_probs_j=policy,
            observed_returns=observed_return,
            behaviour_probs_obs=behaviour,
            mode="signed_policy_contrast",
        )
        plugin = (
            0.7 * 1.0 + 0.2 * 2.0 + 0.1 * 4.0
            - (1.0 + 2.0 + 4.0) / 3.0
        )
        expected = plugin + (0.7 - 1.0 / 3.0) / 0.6 * (1.5 - 1.0)
        wrong_q_minus_observed = plugin + (1.0 / 3.0 - 1.0) / 0.6 * 0.5
        wrong_b_vs_pi = plugin + (1.0 - 0.7 / 0.6) * 0.5
        # This is algebraically correct for the different realised-action
        # contrast psi(A)-E_pi[psi]. It must not replace the V_pi-V_uniform
        # policy-contrast score used by H1.
        realised_action_contrast = (
            1.0 - (0.7 * 1.0 + 0.2 * 2.0 + 0.1 * 4.0)
            + (1.0 - 0.7) / 0.6 * (1.5 - 1.0)
        )
        self.assertAlmostEqual(float(result["effect"][0, 0]), expected, places=6)
        self.assertGreater(abs(expected - wrong_q_minus_observed), 0.1)
        self.assertGreater(abs(expected - wrong_b_vs_pi), 0.1)
        self.assertGreater(abs(expected - realised_action_contrast), 0.1)

    def test_missing_aipw_inputs_is_visible(self):
        proxy = _proxy(effect_mode="signed_policy_contrast", use_dr=True)
        result = proxy._compute_effects(
            torch.zeros((1, 2, 3, 1)), np.array([0, 1]),
            policy_probs_j=np.full((2, 3), 1.0 / 3.0),
            mode="signed_policy_contrast",
        )
        self.assertFalse(result["dr_applied"])
        self.assertEqual(result["dr_applied_rows"], 0)

    def test_invalid_propensity_fails_closed(self):
        proxy = _proxy(effect_mode="signed_policy_contrast", use_dr=True)
        with self.assertRaisesRegex(ValueError, "behaviour_probs_obs"):
            proxy._compute_effects(
                torch.zeros((1, 1, 3, 1)),
                np.array([0]),
                policy_probs_j=np.array([[0.7, 0.2, 0.1]]),
                observed_returns=np.array([0.0]),
                behaviour_probs_obs=np.array([0.0]),
                mode="signed_policy_contrast",
            )

    def test_logged_marginal_propensity_matches_action_frequency(self):
        eps = 0.2
        policy = np.array([0.70, 0.20, 0.10], dtype=np.float32)
        expected = eps / 3.0 + (1.0 - eps) * policy
        policy_rng = np.random.RandomState(11)
        controller = EpsilonForcedActionController(
            n_agents=1,
            action_dim=3,
            eps=eps,
            max_forced_per_step=None,
            rng=np.random.RandomState(12),
        )
        counts = np.zeros(3, dtype=np.int64)
        returned = None
        for _ in range(30000):
            actions = [int(policy_rng.choice(3, p=policy))]
            _, returned = controller.apply(actions, policy.reshape(1, -1))
            counts[actions[0]] += 1

        empirical = counts / counts.sum()
        np.testing.assert_allclose(returned[0], expected, atol=1e-7)
        np.testing.assert_allclose(empirical, expected, atol=0.01)

    def test_tiny_supervised_overfit_proves_action_head_representability(self):
        rng = np.random.RandomState(3)
        proxy = _proxy(
            action_dim=3, effect_mode="signed_policy_contrast", use_dr=False,
            pair_feat_dim=2, lr=5e-3,
        )
        contexts = rng.uniform(-1.0, 1.0, size=(30, 2)).astype(np.float32)
        action_w = np.array([
            [-1.2, 0.4], [0.5, -1.0], [1.4, 0.8]
        ], dtype=np.float32)
        offsets = np.array([-0.3, 0.2, 0.6], dtype=np.float32)

        def outcome(x, action):
            return float(x @ action_w[action] + offsets[action])

        for x in contexts:
            for action in range(3):
                for _ in range(4):
                    y = outcome(x, action)
                    proxy.add_sample(
                        ego_id=0,
                        neighbor_id=1,
                        obs_i=x,
                        action_i=0,
                        observed_action_j=action,
                        z_core_excl_j=np.zeros(1, dtype=np.float32),
                        m_periph_excl_j=np.zeros(1, dtype=np.float32),
                        belief_summary=np.zeros(1, dtype=np.float32),
                        pair_feat=x,
                        target_return_h=y,
                        target_returns_multi=np.array([y], dtype=np.float32),
                        behaviour_prob_j=1.0 / 3.0,
                        was_forced=True,
                    )

        with contextlib.redirect_stdout(io.StringIO()):
            proxy.train_step(n_steps=220, batch_size=96, holdout_size=0)

        scored = proxy.score_batch_full(
            obs_i_batch=contexts,
            action_i_batch=np.zeros(len(contexts), dtype=np.int64),
            observed_action_j_batch=np.zeros(len(contexts), dtype=np.int64),
            z_core_excl_j_batch=np.zeros((len(contexts), 1), dtype=np.float32),
            m_periph_excl_j_batch=np.zeros((len(contexts), 1), dtype=np.float32),
            belief_summary_batch=np.zeros((len(contexts), 1), dtype=np.float32),
            pair_feat_batch=contexts,
            policy_probs_j_batch=np.tile(
                np.array([[0.7, 0.2, 0.1]], dtype=np.float32),
                (len(contexts), 1),
            ),
        )
        truth = np.array([
            (
                0.7 * outcome(x, 0) + 0.2 * outcome(x, 1)
                + 0.1 * outcome(x, 2)
                - np.mean([outcome(x, a) for a in range(3)])
            )
            for x in contexts
        ])
        estimate = scored["mu"]
        rank_corr = float(np.corrcoef(
            np.argsort(np.argsort(truth)), np.argsort(np.argsort(estimate))
        )[0, 1])
        mae = float(np.mean(np.abs(truth - estimate)))
        self.assertGreater(rank_corr, 0.95)
        self.assertLess(mae, 0.12)
        coverage = proxy.get_action_coverage_diagnostics()
        self.assertEqual(coverage["actions_seen"], 3)
        self.assertEqual(coverage["forced_actions_seen"], 3)


class H1ClaimGateTests(unittest.TestCase):
    @staticmethod
    def _supporting_rows():
        epsilon_bias = {
            "plugin_eps000": 0.60,
            "plugin_eps001": 0.50,
            "plugin_eps003": 0.30,
            "plugin_eps005": 0.30,
            "row_aipw_diag_eps005": 0.10,
            "plugin_eps008": 0.08,
            "plugin_eps012": 0.05,
        }
        rows = []
        for seed in range(8):
            for variant, _ in h1_calibration.VARIANTS:
                if variant == "row_aipw_diag_eps005":
                    signed_rank = 0.80 + 0.002 * seed
                    forcing_rate = 0.05
                    heldout_return = 9.0 + 0.01 * seed
                elif variant == "plugin_eps005":
                    signed_rank = 0.40 + 0.002 * seed
                    forcing_rate = 0.05
                    heldout_return = 9.0 + 0.01 * seed
                elif variant == "plugin_eps000":
                    signed_rank = 0.20 + 0.002 * seed
                    forcing_rate = 0.0
                    heldout_return = 10.0 + 0.01 * seed
                else:
                    signed_rank = 0.50
                    forcing_rate = float(variant[-3:]) / 1000.0
                    heldout_return = 9.5
                rows.append({
                    "variant": variant,
                    "seed": seed,
                    "rank_correlation_mean": 0.85,
                    "signed_spearman_mean": signed_rank,
                    "sign_agreement_mean": 0.90,
                    "signed_bias_mean": epsilon_bias[variant],
                    "signed_mae_mean": abs(epsilon_bias[variant]),
                    "range_rank_correlation_mean": 0.30,
                    "q_spearman_mean": 0.70,
                    "q_mae_mean": 0.10,
                    "q_rmse_mean": 0.15,
                    "capacity_rank_correlation_mean": 0.85,
                    "capacity_mae_mean": 0.10,
                    "capacity_bias_mean": 0.02,
                    "capacity_active_mae_mean": 0.08,
                    "capacity_active_spearman_mean": 0.75,
                    "capacity_null_fpr_mean": 0.05,
                    "oracle_core_f1_mean": 0.80,
                    "direction_spearman_mean": signed_rank,
                    "direction_mae_mean": abs(epsilon_bias[variant]),
                    "direction_bias_mean": epsilon_bias[variant],
                    "direction_sign_agreement_mean": 0.90,
                    "direction_active_mae_mean": 0.08,
                    "direction_active_spearman_mean": 0.72,
                    "direction_active_sign_agreement_mean": 0.90,
                    "direction_null_fpr_mean": 0.05,
                    "direction_row_aipw_signed_spearman_mean": (
                        0.80 + 0.002 * seed
                    ),
                    "direction_row_aipw_signed_mae_mean": 0.10,
                    "direction_row_aipw_sign_agreement_mean": 0.90,
                    # Missing eps=0 coverage is an intended control outcome;
                    # only the main arm enters the scientific coverage gate.
                    "action_coverage_gate_pass": variant == "plugin_eps005",
                    "dr_clipping_absent": True,
                    "realised_forcing_rate": forcing_rate,
                    "heldout_policy_return_mean_per_agent": heldout_return,
                    "policy_return_endpoint_measured": True,
                })
        return rows

    def test_full_gate_requires_q_c_d_recovery_and_support_integrity(self):
        claim = h1_calibration._claim_gate(self._supporting_rows())
        self.assertTrue(claim["h1_claim_gate_pass"])
        self.assertTrue(claim["h1_main_action_coverage_gate_pass"])
        self.assertTrue(claim["h1a_q_recovery_pass"])
        self.assertTrue(claim["h1b_capacity_recovery_pass"])
        self.assertTrue(claim["h1c_direction_recovery_pass"])
        self.assertTrue(claim["h1_support_integrity_pass"])
        self.assertTrue(claim["h1_exp1_reporting_complete"])
        self.assertGreater(
            claim[
                "h1_estimator_ablation"
            ][
                "direction_rank_row_aipw_minus_plugin_paired_bootstrap"
            ]["ci95_low"],
            0.0,
        )
        self.assertEqual(
            claim["h1_forcing_reporting"]
            ["forcing_return_cost_paired_bootstrap"]
            ["paired_seed_differences"][0]["difference"],
            1.0,
        )

    def test_failed_capacity_recovery_blocks_h1(self):
        rows = self._supporting_rows()
        for row in rows:
            if row["variant"] == "plugin_eps005":
                row["capacity_active_spearman_mean"] = 0.10
        claim = h1_calibration._claim_gate(rows)
        self.assertFalse(claim["h1_claim_gate_pass"])
        self.assertFalse(claim["h1b_capacity_recovery_pass"])

    def test_missing_forcing_return_endpoint_is_reported_separately(self):
        rows = self._supporting_rows()
        for row in rows:
            if row["variant"] == "plugin_eps000":
                row["policy_return_endpoint_measured"] = False
        claim = h1_calibration._claim_gate(rows)
        self.assertFalse(claim["h1_exp1_reporting_complete"])
        self.assertTrue(claim["h1_claim_gate_pass"])


if __name__ == "__main__":
    unittest.main()
