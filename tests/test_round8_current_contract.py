import copy

import numpy as np
import pytest
import torch

from models.crossfit_aipw import proxy_context_features
from models.drift_probe import DriftDetector
from models.intervention import EpsilonForcedActionController
from models.structural_proxy import LocalCounterfactualProxyEnsemble


def _proxy(*, action_dim=3, n_horizons=1, use_dr=False, iw_clip=20.0):
    return LocalCounterfactualProxyEnsemble(
        obs_dim=2,
        action_dim=action_dim,
        core_dim=1,
        periph_dim=1,
        belief_dim=1,
        pair_feat_dim=0,
        n_ensemble=1,
        n_horizons=n_horizons,
        effect_mode="signed_policy_contrast",
        use_doubly_robust=use_dr,
        iw_clip=iw_clip,
        use_vmap_ensemble=False,
        device="cpu",
    )


def test_targeted_forcing_preserves_budget_after_saturation():
    controller = EpsilonForcedActionController(
        n_agents=4,
        action_dim=3,
        eps=0.8,
        max_forced_per_step=None,
        rng=np.random.RandomState(7),
    )
    controller.set_priority(np.asarray([100.0, 1.0, 1.0, 1.0]))
    eps = controller.get_eps_per_agent()
    assert np.all(eps > 0.0)
    assert np.all(eps <= 1.0)
    assert float(np.mean(eps)) == pytest.approx(0.8, abs=1e-10)
    assert float(np.max(eps)) == pytest.approx(1.0, abs=1e-10)


def test_policy_contrast_rejects_target_policy_mass_on_invalid_actions():
    proxy = _proxy(action_dim=3)
    preds = torch.zeros((1, 1, 3, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="zero mass to invalid actions"):
        proxy._compute_effects(
            preds,
            np.asarray([0]),
            policy_probs_j=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            valid_action_mask=np.asarray([[1, 1, 0]], dtype=bool),
            mode="signed_policy_contrast",
        )


def test_truncated_online_aipw_clips_full_signed_coefficient():
    proxy = _proxy(action_dim=2, use_dr=True, iw_clip=20.0)
    preds = torch.zeros((1, 1, 2, 1), dtype=torch.float32)
    out = proxy._compute_effects(
        preds,
        np.asarray([0]),
        policy_probs_j=np.asarray([[0.6, 0.4]], dtype=np.float32),
        observed_returns=np.asarray([1.0], dtype=np.float32),
        behaviour_probs_obs=np.asarray([0.001], dtype=np.float32),
        mode="signed_policy_contrast",
    )
    # q(0)=0.5, so raw signed coefficient is (0.6-0.5)/0.001=100.
    # Canonical truncated AIPW clips that complete coefficient to +20.
    assert float(out["effect"][0, 0]) == pytest.approx(20.0, abs=1e-5)
    assert out["dr_clipped_rows"] == 1


def test_multi_lag_proxy_rejects_fabricated_flat_latency_target():
    proxy = _proxy(action_dim=2, n_horizons=3)
    with pytest.raises(ValueError, match="cannot be duplicated across lags"):
        proxy.add_sample(
            ego_id=0,
            neighbor_id=1,
            obs_i=np.zeros(2, dtype=np.float32),
            action_i=0,
            observed_action_j=0,
            z_core_excl_j=np.zeros(1, dtype=np.float32),
            m_periph_excl_j=np.zeros(1, dtype=np.float32),
            belief_summary=np.zeros(1, dtype=np.float32),
            target_return_h=1.0,
        )


def test_crossfit_context_is_partition_independent_and_fails_closed_without_raw_context():
    base = {
        "action_i": 0,
        "obs_i": np.asarray([1.0, 2.0], dtype=np.float32),
        "pair_feat": np.asarray([0.25], dtype=np.float32),
        "context_items": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "context_mask": np.asarray([1.0, 1.0], dtype=np.float32),
        "z_core_excl_j": np.asarray([999.0], dtype=np.float32),
        "m_periph_excl_j": np.asarray([-999.0], dtype=np.float32),
    }
    changed_downstream = copy.deepcopy(base)
    changed_downstream["z_core_excl_j"] = np.asarray([-1234.0], dtype=np.float32)
    changed_downstream["m_periph_excl_j"] = np.asarray([5678.0], dtype=np.float32)
    np.testing.assert_allclose(
        proxy_context_features(base, 2),
        proxy_context_features(changed_downstream, 2),
    )
    missing_raw = copy.deepcopy(base)
    missing_raw.pop("context_items")
    missing_raw.pop("context_mask")
    with pytest.raises(ValueError, match="partition-independent"):
        proxy_context_features(missing_raw, 2)


def _drift_sample(episode, value=0.0):
    return {
        "obs_i": np.asarray([0.1], dtype=np.float32),
        "action_i": 0,
        "observed_action_j": 1,
        "target_lag_rewards": np.asarray([value], dtype=np.float32),
        "horizon_complete": True,
        "episode_id": int(episode),
    }


def test_cusum_calibration_rejects_mixed_monitoring_horizons():
    with pytest.raises(ValueError, match="one frozen monitoring horizon"):
        DriftDetector.calibrate_cusum_from_no_change(
            [[0.0, 0.1, -0.1], [0.0, 0.1]],
            allowance=0.5,
            false_alarm_target=0.05,
        )


def test_drift_monitoring_uses_same_batch_size_as_reference_statistic():
    detector = DriftDetector(
        obs_dim=1, action_dim=2, n_horizons=1,
        warmup_batches=1, batch_size=1, window=5, seed=3,
    )
    detector.snapshot(episode=0)
    with pytest.raises(ValueError, match="must equal the calibrated batch_size"):
        detector.measure([_drift_sample(0)], n=2)


def test_post_trigger_live_witness_training_is_restricted_to_new_regime():
    detector = DriftDetector(
        obs_dim=1, action_dim=2, n_horizons=1,
        warmup_batches=1, batch_size=1, window=5,
        recalibrate_after=2, seed=9,
    )
    stable = [_drift_sample(0, 0.1 * i) for i in range(6)]
    detector.prepare_for_monitoring(stable, episode=0, reference_batches=5)
    detector.notify_trigger(episode=1)
    calls = []
    original = detector.train_batches

    def spy(buffer, n_batches=1, min_episode_id=None):
        calls.append(min_episode_id)
        return original(buffer, n_batches, min_episode_id=min_episode_id)

    detector.train_batches = spy
    detector.step(episode=2, buffer=stable, n_train_batches=1)
    assert calls and calls[-1] == 2
