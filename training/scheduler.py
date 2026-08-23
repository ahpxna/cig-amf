"""Two-timescale scheduling, shift detection, and uncertainty inflation.

Version 2 differs from version 1 in three ways:
  1. It uses a frozen-probe Page-CUSUM trigger and records influence-matrix
     movement as a diagnostic.
  2. A trigger both accelerates updates and INFLATES belief uncertainty. The
     core then contracts, keeping the system conservative while the new regime
     remains poorly understood.
  3. A refractory interval prevents repeated triggers during adaptation;
     otherwise every core change could trigger the scheduler again.
"""

from typing import Dict, List, Optional

import numpy as np


class TwoTimescaleScheduler:
    """Preserve the v1 API while adding ``evaluate_drift``.

    Existing runners can continue to call ``step_episode``,
    ``should_update_graph``, ``in_warmup``, and ``get_status``.

    Args:
        k0_warmup: Number of Stage 0 episodes.
        alpha_slow_ratio: 0.05 schedules a structure update about every 20 episodes.
        accel_factor / accel_duration: Magnitude and duration of acceleration.
        z_threshold: Shift threshold. Values from 2.5 to 3.0 are reasonable;
            z=3 means three standard deviations from recent behavior.
        require_both: Deprecated compatibility field; matrix movement is not
            part of the final trigger.
        refractory: Number of refractory episodes after each trigger.
        inflation_factor: Multiplicative sigma inflation at a trigger.
    """

    def __init__(
        self,
        k0_warmup: int = 20,
        alpha_fast: float = 1e-3,
        alpha_slow_ratio: float = 0.05,
        accel_factor: float = 4.0,
        accel_duration: int = 8,
        z_threshold: float = 3.0,
        require_both: bool = False,
        refractory: int = 10,
        inflation_factor: float = 2.5,
        inflation_t_reset: int = 1,
    ):
        self.k0_warmup = int(k0_warmup)
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow_ratio = float(alpha_slow_ratio)
        self.alpha_slow_base = self.alpha_fast * self.alpha_slow_ratio
        self.alpha_slow = 0.0

        self.accel_factor = float(accel_factor)
        self.accel_duration = int(accel_duration)

        self.z_threshold = float(z_threshold)
        self.require_both = bool(require_both)
        self.refractory = int(refractory)

        self.inflation_factor = float(inflation_factor)
        self.inflation_t_reset = int(inflation_t_reset)

        self.episode = 0
        self.stage = 0

        self.accel_remaining = 0
        self.trigger_count = 0
        self.last_trigger_episode = None
        self.trigger_log: List[Dict] = []

        # Raw residual history is used only by the backward-compatible v1
        # ``record_structural_residual()`` wrapper. Keep it separate from the
        # state used by ``evaluate_drift()`` so the mechanisms cannot mix.
        self._residual_history: List[float] = []
        self._residual_window = 20

    # ------------------------------------------------------------------
    # API v1
    # ------------------------------------------------------------------

    def in_warmup(self) -> bool:
        return self.stage == 0

    def in_learned_stage(self) -> bool:
        return self.stage == 1

    def force_learned_stage(self):
        """Force Stage 1 for the NoTwoTimescale ablation."""
        self.stage = 1
        self.alpha_slow = self.alpha_slow_base
        self.accel_remaining = 0

    def step_episode(self):
        self.episode += 1

        if self.stage == 0 and self.episode >= self.k0_warmup:
            self.stage = 1
            self.alpha_slow = self.alpha_slow_base

        if self.accel_remaining > 0:
            self.accel_remaining -= 1

            if self.accel_remaining == 0:
                self.alpha_slow = self.alpha_slow_base

    def _base_freq(self) -> int:
        return max(1, int(round(1.0 / max(1e-8, self.alpha_slow_ratio))))

    def _accel_freq(self) -> int:
        r = max(1e-8, self.alpha_slow_ratio * max(1.0, self.accel_factor))
        return max(1, int(round(1.0 / r)))

    def should_update_graph(self) -> bool:
        """Return whether proxy training and belief/core updates are due.

        This method MUST NOT gate replay collection. Replay is collected on
        EVERY episode, including Stage 0.
        """
        if self.stage == 0:
            return False

        freq = (
            self._accel_freq() if self.accel_remaining > 0 else self._base_freq()
        )

        return (self.episode % freq) == 0

    # ------------------------------------------------------------------
    # Shift detection
    # ------------------------------------------------------------------

    def _in_refractory(self) -> bool:
        if self.last_trigger_episode is None:
            return False

        return (self.episode - self.last_trigger_episode) < self.refractory

    def evaluate_drift(
        self,
        probe_z: float = 0.0,
        matrix_z: float = 0.0,
        belief_modules: Optional[Dict] = None,
        drift_detector=None,
    ) -> Dict:
        """Evaluate the frozen-witness Page-CUSUM trigger.

        Args:
            probe_z: Z-score from ``DriftDetector.residual_z_score()``.
            matrix_z: Z-score from ``MatrixDriftDetector.z_score()``.
            belief_modules: ``{ego_id: BayesLightBeliefState}`` to inflate.
            drift_detector: Detector notified to resnapshot after adaptation.

        Returns:
            Status dictionary containing the ``fired`` key.
        """
        out = {
            "episode": int(self.episode),
            "probe_z": float(probe_z),
            "matrix_z": float(matrix_z),
            "fired": False,
            "reason": None,
            "n_inflated": 0,
        }

        if self.stage == 0:
            out["reason"] = "warmup"
            return out

        if self._in_refractory():
            out["reason"] = "refractory"
            return out

        hit_probe = float(probe_z) > self.z_threshold
        # Matrix movement remains a plotted diagnostic/ablation. It is not
        # combined with the prespecified frozen-witness trigger.
        fired = hit_probe

        if not fired:
            out["reason"] = "below_threshold"
            return out

        # ---- TRIGGER ----------------------------------------------------
        self.trigger_count += 1
        self.last_trigger_episode = self.episode

        # (a) Accelerate structure updates.
        self.alpha_slow = self.alpha_slow_base * self.accel_factor
        self.accel_remaining = self.accel_duration

        # (b) Inflate uncertainty so the core contracts conservatively.
        n_inflated = 0

        if belief_modules is not None:
            for mod in belief_modules.values():
                if hasattr(mod, "inflate_uncertainty"):
                    st = mod.inflate_uncertainty(
                        factor=self.inflation_factor,
                        t_reset=self.inflation_t_reset,
                    )
                    n_inflated += int(st["n_pairs_inflated"])

        # (c) Schedule a new probe snapshot; otherwise the alarm persists.
        if drift_detector is not None and hasattr(drift_detector, "notify_trigger"):
            drift_detector.notify_trigger(self.episode)

        out.update({
            "fired": True,
            "reason": "probe_cusum",
            "n_inflated": int(n_inflated),
        })

        self.trigger_log.append(dict(out))

        return out

    # ------------------------------------------------------------------
    # Backward compatibility with the v1 API
    # ------------------------------------------------------------------

    def _residual_z_score(self, residual: float) -> float:
        """Convert a raw residual to a recent-window z-score.

        This history is self-managed and separate from
        DriftDetector/MatrixDriftDetector.
        """
        self._residual_history.append(float(residual))

        if len(self._residual_history) > 500:
            del self._residual_history[:-500]

        h = self._residual_history

        if len(h) < 5:
            return 0.0

        recent = np.asarray(h[-self._residual_window:], dtype=np.float64)
        base = recent[:-1]

        if base.size < 3:
            return 0.0

        mu, sd = float(np.mean(base)), float(np.std(base))

        if sd < 1e-9:
            return 0.0

        return float((h[-1] - mu) / sd)

    def record_structural_residual(self, residual: float) -> bool:
        """Backward-compatible wrapper for runners that use the v1 API.

        ``final_runner.py`` and ``baseline_runner.py`` historically supplied
        one raw residual rather than the two independent z-scores consumed by
        ``evaluate_drift()``. This wrapper derives an internal z-score and
        treats it as the probe trigger with ``matrix_z=0.0``. Runners with real
        DriftDetector and MatrixDriftDetector instances should call
        ``evaluate_drift()`` directly.

        Returns:
            True when the trigger fires, equivalent to v1 ``triggered``.
        """
        z = self._residual_z_score(residual)
        out = self.evaluate_drift(probe_z=z, matrix_z=0.0)

        return bool(out["fired"])

    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        return {
            "episode": int(self.episode),
            "stage": int(self.stage),
            "alpha_fast": float(self.alpha_fast),
            "alpha_slow": float(self.alpha_slow),
            "accel_remaining": int(self.accel_remaining),
            "trigger_count": int(self.trigger_count),
            "last_trigger_episode": self.last_trigger_episode,
            "in_refractory": bool(self._in_refractory()),
            "z_threshold": float(self.z_threshold),
        }
