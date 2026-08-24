import copy

import numpy as np

from models.drift_probe import DriftDetector


def _sample(episode_id, reward=0.0):
    return {
        "obs_i": np.asarray([0.25], dtype=np.float32),
        "action_i": 0,
        "observed_action_j": 1,
        "target_lag_rewards": np.asarray([reward], dtype=np.float32),
        "horizon_complete": True,
        "episode_id": int(episode_id),
    }


def _detector():
    return DriftDetector(
        obs_dim=1,
        action_dim=2,
        n_horizons=1,
        discount=1.0,
        warmup_batches=1,
        batch_size=1,
        window=5,
        recalibrate_after=1,
        seed=7,
    )


def test_residual_z_score_does_not_bootstrap_reference_from_monitoring_history():
    detector = _detector()
    detector.snapshot(episode=0)
    detector.residual_history.extend([1.0] * detector.window)

    assert detector.residual_z_score() == 0.0
    assert detector.reference_mean is None
    assert detector.reference_std is None
    assert detector.reference_sample_count == 0
    assert not detector.is_monitoring_ready()


def test_post_trigger_recalibration_waits_for_post_trigger_replay():
    detector = _detector()
    stable = [_sample(0, reward=float(i) / 10.0) for i in range(6)]
    detector.prepare_for_monitoring(stable, episode=0, reference_batches=5)
    assert detector.is_monitoring_ready()

    detector.notify_trigger(episode=1)
    old_frozen = copy.deepcopy(detector.frozen.state_dict())

    waiting = detector.step(episode=2, buffer=stable, n_train_batches=0)
    assert waiting["phase"] == "reference_recalibration"
    assert detector.pending_recalibration_at == 2
    assert detector.recalibration_reference_min_episode == 2
    assert detector.is_monitoring_ready()
    for key, value in old_frozen.items():
        np.testing.assert_allclose(
            detector.frozen.state_dict()[key].detach().cpu().numpy(),
            value.detach().cpu().numpy(),
        )

    post_shift = stable + [_sample(2, reward=0.5 + float(i) / 10.0) for i in range(6)]
    recalibrated = detector.step(episode=3, buffer=post_shift, n_train_batches=0)
    assert recalibrated["phase"] == "recalibrated"
    assert detector.pending_recalibration_at is None
    assert detector.recalibration_reference_min_episode is None
    assert detector.reference_sample_count >= detector.window
    assert detector.is_monitoring_ready()
