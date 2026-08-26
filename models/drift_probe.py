"""
drift_probe.py — detect structural shifts with a frozen witness.

V1 DEFECT: A SELF-CONTAMINATING RESIDUAL

v1 used the continuously trained proxy residual e_t=|R_hat-R_actual| as the
structural-shift trigger. This failed in two ways:

1. When the world changed, the proxy silently learned the new regime, became
   accurate again, and kept its residual small. It absorbed the event it was
   supposed to detect and never raised an alarm.
2. Proxy inputs included Z^{-j} and M^{-j}, both dependent on the current
   partition. A partition change altered inputs, raised residual, fired the
   trigger, and changed the partition again. The trigger responded to its own
   internal motion rather than to the external world.

SOLUTION: A FROZEN, BLINDED WITNESS

The probe is analogous to a diner returning after five years: a daily diner
gradually adapts to a new chef and may not notice the change, whereas the
returning diner's taste remains anchored in the past. The probe has two
properties:

  FROZEN: after training, its weights are locked. Its memory remains anchored
      at snapshot time rather than drifting with the new regime.
  BLINDED/CONTEXT-FREE: it observes only (o_i,a_i,a_j), never Z, M, or B.
      Partition changes cannot affect it, eliminating the second contamination
      path completely.

Its residual is a frozen regime-change witness: it is insulated from internal
partition bookkeeping, but policy drift or covariate-support shifts may still
raise it without a structural-law change. The H2 protocol measures that false
trigger rate under behavioural-only intervention.

IMPORTANT: SNAPSHOT AGAIN AFTER ADAPTATION

After a true structural shift, the old probe remains permanently wrong because
the world is new and its memory is old. Without a new snapshot it alarms
forever. The required cycle is: trigger, adapt, wait `recalibrate_after`, then
take a new snapshot. `maybe_recalibrate()` implements this cycle.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralDriftProbe(nn.Module):
    """
    Small network predicting ego return only from (o_i,a_i,a_j).

    It is deliberately weak and blinded to anything partition-dependent. It
    need not predict exceptionally well; its error is treated as a regime-
    change witness and is explicitly checked for behavioural false alarms.
    obs_dim/action_dim match the main proxy,
    hidden should remain small (64) because larger models overfit and make the
    residual noisy, and n_horizons matches the main proxy.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 64,
        n_horizons: int = 3,
        discount: float = 0.97,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_horizons = int(n_horizons)
        self.discount = float(discount)

        in_dim = self.obs_dim + 2 * self.action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), self.n_horizons),
        )

    def forward(
        self,
        obs_i: torch.Tensor,            # [B, obs_dim]
        action_i_onehot: torch.Tensor,  # [B, action_dim]
        action_j_onehot: torch.Tensor,  # [B, action_dim]
    ) -> torch.Tensor:
        """Returns: [B, n_horizons]"""
        x = torch.cat([obs_i, action_i_onehot, action_j_onehot], dim=-1)
        return self.net(x)


class DriftDetector:
    """
    Manage the probe lifecycle: train, freeze, measure, and resnapshot.

    Runner usage:
        det = DriftDetector(obs_dim, action_dim, n_horizons, device)
        det.step(episode, proxy.buffer)
        fired = det.residual_z_score() > threshold

    warmup_batches is the number of batches before the first freeze;
    recalibrate_after is the post-trigger episode delay before resnapshotting;
    window normalizes residuals into z-scores.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_horizons: int = 3,
        hidden: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        warmup_batches: int = 200,
        batch_size: int = 256,
        recalibrate_after: int = 15,
        window: int = 20,
        discount: float = 0.97,
        cusum_allowance: float = 0.5,
        cusum_threshold: float = 8.0,
        seed: int = 0,
    ):
        torch.manual_seed(int(seed))

        self.device = device
        self.action_dim = int(action_dim)
        self.n_horizons = int(n_horizons)
        self.discount = float(discount)
        self.batch_size = int(batch_size)
        self.warmup_batches = int(warmup_batches)
        self.recalibrate_after = int(recalibrate_after)
        self.window = int(window)
        self.cusum_allowance = float(cusum_allowance)
        self.cusum_threshold = float(cusum_threshold)
        self.reference_mean = None
        self.reference_std = None
        self.cusum_stat = 0.0
        self.latest_standardized_residual = 0.0

        self.live = StructuralDriftProbe(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=hidden,
            n_horizons=n_horizons,
            discount=discount,
        ).to(device)

        self.optim = torch.optim.Adam(self.live.parameters(), lr=float(lr))

        # Frozen baseline. None until warm-up is complete.
        self.frozen: Optional[StructuralDriftProbe] = None

        self.rng = np.random.RandomState(int(seed))

        self.n_batches_trained = 0
        self.residual_history: List[float] = []
        self.last_snapshot_episode = None
        self.pending_recalibration_at = None
        self.recalibration_reference_min_episode = None
        self.n_snapshots = 0
        self.reference_sample_count = 0

    # ------------------------------------------------------------------

    def _one_hot(self, a: np.ndarray) -> torch.Tensor:
        raw = np.asarray(a)
        if raw.ndim != 1 or not np.all(np.isfinite(raw)):
            raise ValueError("drift probe action identities must be finite scalars")
        if not np.all(raw == np.floor(raw)):
            raise ValueError("drift probe action identities must be integers")
        raw = raw.astype(np.int64, copy=False)
        if np.any(raw < 0) or np.any(raw >= self.action_dim):
            raise ValueError(
                "drift probe action identity is outside the action space: "
                f"expected [0, {self.action_dim})"
            )
        t = torch.tensor(raw, dtype=torch.long, device=self.device)

        return F.one_hot(t, num_classes=self.action_dim).to(dtype=torch.float32)

    def _batch(self, buffer, n: int, min_episode_id: Optional[int] = None):
        """Get a complete-horizon batch, optionally restricted to a new regime."""
        if buffer is None or len(buffer) == 0:
            return None

        # Page-CUSUM targets a complete discounted H-return.  Right-censored
        # episode tails are supervised by the proxy through lag masks, but are
        # not valid witness observations.
        buf = [
            sample for sample in buffer
            if bool(sample.get("horizon_complete", True))
            and (
                min_episode_id is None
                or sample.get("episode_id") is not None
                and int(sample["episode_id"]) >= int(min_episode_id)
            )
        ]
        if not buf:
            return None
        n = int(min(n, len(buf)))
        idx = self.rng.choice(len(buf), size=n, replace=False)
        batch = [buf[i] for i in idx]

        obs = torch.tensor(
            np.stack([b["obs_i"] for b in batch], axis=0),
            dtype=torch.float32, device=self.device,
        )                                                    # [B, obs_dim]
        a_i = self._one_hot([b["action_i"] for b in batch])  # [B, A]
        a_j = self._one_hot(
            [b["observed_action_j"] for b in batch]
        )                                                    # [B, A]
        tgt = torch.tensor(
            np.stack([self._target_lag_rewards(b) for b in batch], axis=0),
            dtype=torch.float32, device=self.device,
        )                                                    # [B, H]

        return obs, a_i, a_j, tgt

    def _target_lag_rewards(self, sample) -> np.ndarray:
        """Return direct-lag rewards and convert legacy cumulative targets."""
        direct = sample.get("target_lag_rewards")
        if direct is not None:
            values = np.asarray(direct, dtype=np.float32).reshape(-1)
        else:
            cumulative = sample.get("target_returns_multi")
            if cumulative is None:
                raise KeyError("drift replay sample has no lag or cumulative target")
            cumulative = np.asarray(cumulative, dtype=np.float32).reshape(-1)
            values = np.empty_like(cumulative)
            if cumulative.size:
                values[0] = cumulative[0]
                for lag in range(1, cumulative.size):
                    scale = self.discount ** lag
                    values[lag] = (
                        cumulative[lag] - cumulative[lag - 1]
                    ) / max(scale, 1e-12)
        if values.size != self.n_horizons:
            raise ValueError(
                "drift target horizon mismatch: "
                f"expected {self.n_horizons}, received {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("drift target contains NaN or infinity")
        return values

    def _discounted_return_residual(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-row absolute error of the reconstructed discounted H-return."""
        expected = (pred.shape[0], self.n_horizons)
        if tuple(pred.shape) != expected or tuple(target.shape) != expected:
            raise ValueError(
                "drift prediction/target shape mismatch: "
                f"pred={tuple(pred.shape)}, target={tuple(target.shape)}, "
                f"expected={expected}"
            )
        weights = torch.pow(
            torch.as_tensor(
                self.discount, dtype=pred.dtype, device=pred.device
            ),
            torch.arange(
                self.n_horizons, dtype=pred.dtype, device=pred.device
            ),
        )
        pred_return = torch.sum(pred * weights.view(1, -1), dim=1)
        target_return = torch.sum(target * weights.view(1, -1), dim=1)
        return torch.abs(pred_return - target_return)

    # ------------------------------------------------------------------

    def train_batches(self, buffer, n_batches: int = 1) -> float:
        """Train live (frozen never touched) model."""
        losses = []

        for _ in range(int(n_batches)):
            got = self._batch(buffer, self.batch_size)

            if got is None:
                break

            obs, a_i, a_j, tgt = got

            self.live.train()
            pred = self.live(obs, a_i, a_j)          # [B, H]
            if pred.shape != tgt.shape:
                raise ValueError(
                    "drift training horizon mismatch: "
                    f"pred={tuple(pred.shape)}, target={tuple(tgt.shape)}"
                )
            loss = F.mse_loss(pred, tgt)

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.live.parameters(), 1.0)
            self.optim.step()

            losses.append(float(loss.detach().cpu().item()))
            self.n_batches_trained += 1

        return float(np.mean(losses)) if losses else 0.0

    def snapshot(self, episode: Optional[int] = None):
        """Snapshot the live model into a frozen witness with fixed memory."""
        import copy

        self.frozen = copy.deepcopy(self.live).to(self.device)

        for p in self.frozen.parameters():
            p.requires_grad_(False)

        self.frozen.eval()

        self.last_snapshot_episode = episode
        self.pending_recalibration_at = None
        self.n_snapshots += 1

        # A new witness invalidates the old residual scale.
        self.residual_history.clear()
        self.reference_mean = None
        self.reference_std = None
        self.reference_sample_count = 0
        self.cusum_stat = 0.0
        self.latest_standardized_residual = 0.0

    def measure(self, buffer, n: int = 512) -> Optional[float]:
        """
        Evaluate the frozen model on the newest data.

        Draw from the end of the buffer rather than randomly because the
        question is whether the world changed recently. Return the mean
        absolute error of the reconstructed discounted H-step return.
        """
        if self.frozen is None or buffer is None or len(buffer) == 0:
            return None

        buf = [sample for sample in list(buffer)[-int(n):]
               if bool(sample.get("horizon_complete", True))]
        if not buf:
            return None

        obs = torch.tensor(
            np.stack([b["obs_i"] for b in buf], axis=0),
            dtype=torch.float32, device=self.device,
        )
        a_i = self._one_hot([b["action_i"] for b in buf])
        a_j = self._one_hot([b["observed_action_j"] for b in buf])
        tgt = torch.tensor(
            np.stack([self._target_lag_rewards(b) for b in buf], axis=0),
            dtype=torch.float32, device=self.device,
        )

        with torch.no_grad():
            pred = self.frozen(obs, a_i, a_j)                  # [B, H]
            res = torch.mean(self._discounted_return_residual(pred, tgt))

        val = float(res.cpu().item())
        self.residual_history.append(val)

        if len(self.residual_history) > 500:
            del self.residual_history[:-500]

        return val

    def calibrate_reference(
        self, buffer, n_batches: Optional[int] = None, batch_size: Optional[int] = None,
        min_episode_id: Optional[int] = None,
    ) -> Dict:
        """Calibrate the fixed Page-CUSUM reference on stable replay only."""
        if self.frozen is None:
            raise RuntimeError("cannot calibrate drift reference before snapshot")
        if buffer is None or len(buffer) == 0:
            raise RuntimeError("cannot calibrate drift reference from empty replay")
        count = int(self.window if n_batches is None else n_batches)
        size = int(self.batch_size if batch_size is None else batch_size)
        if count < max(5, self.window):
            raise ValueError(
                f"reference calibration needs at least {max(5, self.window)} batches"
            )
        values = []
        for _ in range(count):
            got = self._batch(buffer, size, min_episode_id=min_episode_id)
            if got is None:
                break
            obs, a_i, a_j, target = got
            with torch.no_grad():
                pred = self.frozen(obs, a_i, a_j)
                residual = torch.mean(
                    self._discounted_return_residual(pred, target)
                )
            values.append(float(residual.detach().cpu().item()))
        if len(values) < max(5, self.window):
            raise RuntimeError("insufficient stable residual batches for calibration")
        reference = np.asarray(values, dtype=np.float64)
        self.reference_mean = float(np.mean(reference))
        self.reference_std = float(max(np.std(reference), 1e-9))
        self.reference_sample_count = int(reference.size)
        self.residual_history.clear()
        self.cusum_stat = 0.0
        self.latest_standardized_residual = 0.0
        return {
            "reference_mean": self.reference_mean,
            "reference_std": self.reference_std,
            "reference_sample_count": self.reference_sample_count,
        }

    def prepare_for_monitoring(
        self, buffer, episode: Optional[int] = None, reference_batches: Optional[int] = None
    ) -> Dict:
        """Train, freeze, and calibrate entirely on a stable pre-change replay."""
        remaining = max(0, self.warmup_batches - self.n_batches_trained)
        if remaining:
            self.train_batches(buffer, remaining)
        if self.n_batches_trained < self.warmup_batches:
            raise RuntimeError(
                "drift witness did not reach its required training budget"
            )
        self.snapshot(episode)
        calibration = self.calibrate_reference(
            buffer, n_batches=reference_batches
        )
        return {
            "frozen_ready": True,
            "n_batches_trained": int(self.n_batches_trained),
            **calibration,
        }

    def is_monitoring_ready(self) -> bool:
        return bool(
            self.frozen is not None
            and self.reference_mean is not None
            and self.reference_std is not None
            and self.reference_sample_count >= max(5, self.window)
        )

    def residual_z_score(self) -> float:
        """Standardize only after an explicit pre-monitoring calibration."""
        if not self.is_monitoring_ready() or not self.residual_history:
            self.latest_standardized_residual = 0.0
            return 0.0
        h = self.residual_history
        z = float(
            (h[-1] - self.reference_mean) / self.reference_std
        )
        self.latest_standardized_residual = z
        self.cusum_stat = max(
            0.0, self.cusum_stat + z - self.cusum_allowance
        )
        # Keep the raw standardized residual and the Page-CUSUM statistic
        # separate.  Calibration consumes z-trajectories; the scheduler
        # consumes the accumulated CUSUM statistic.
        return float(z)

    @staticmethod
    def page_cusum_maxima(z_sequences, allowance: float):
        """Return one run-level Page-CUSUM maximum per no-change sequence.

        This pure helper is used by the development-only calibration tool.  It
        keeps the false-alarm threshold tied to the exact standardized
        residual process, rather than treating an arbitrary z threshold as a
        scientific hyperparameter.
        """
        maxima = []
        allowance = float(allowance)
        for sequence in z_sequences:
            statistic = 0.0
            maximum = 0.0
            for value in np.asarray(sequence, dtype=np.float64).reshape(-1):
                if not np.isfinite(value):
                    raise ValueError("CUSUM calibration contains non-finite z")
                statistic = max(0.0, statistic + float(value) - allowance)
                maximum = max(maximum, statistic)
            maxima.append(maximum)
        if not maxima:
            raise ValueError("CUSUM calibration requires at least one trajectory")
        return np.asarray(maxima, dtype=np.float64)

    @classmethod
    def calibrate_cusum_from_no_change(
        cls, z_sequences, allowance: float, false_alarm_target: float
    ):
        """Derive a frozen run-level threshold from no-change trajectories."""
        alpha = float(false_alarm_target)
        if not 0.0 < alpha < 1.0:
            raise ValueError("false_alarm_target must lie strictly between zero and one")
        maxima = cls.page_cusum_maxima(z_sequences, allowance)
        threshold = float(np.quantile(maxima, 1.0 - alpha, method="higher"))
        observed = float(np.mean(maxima > threshold))
        return {
            "cusum_allowance": float(allowance),
            "cusum_threshold": threshold,
            "target_false_alarm_rate": alpha,
            "observed_false_alarm_rate": observed,
            "n_no_change_trajectories": int(maxima.size),
            "monitoring_horizon": int(max(len(np.asarray(v).reshape(-1)) for v in z_sequences)),
            "maxima": maxima.tolist(),
        }

    # ------------------------------------------------------------------

    def step(self, episode: int, buffer, n_train_batches: int = 5) -> Dict:
        """Run the complete lifecycle once per episode and return log state."""
        # Train normally before the first snapshot.
        if self.frozen is None:
            self.train_batches(buffer, n_train_batches)

            if self.n_batches_trained >= self.warmup_batches:
                self.snapshot(episode)
                try:
                    self.calibrate_reference(
                        buffer, n_batches=max(20, int(self.window))
                    )
                except RuntimeError:
                    return {
                        "phase": "reference_calibration",
                        "batches": int(self.n_batches_trained),
                        "residual": None,
                        "z": 0.0,
                    }
                return {
                    "phase": "monitoring_ready",
                    "batches": int(self.n_batches_trained),
                    "residual": None,
                    "z": 0.0,
                }

            return {
                "phase": "warmup",
                "batches": int(self.n_batches_trained),
                "residual": None,
                "z": 0.0,
            }

        # After freezing, the live model keeps training for the next snapshot.
        # The frozen witness is never updated.
        self.train_batches(buffer, n_train_batches)

        # Re-freeze only after the adaptation delay, then calibrate the new
        # witness exclusively on post-trigger samples before monitoring resumes.
        if (
            self.pending_recalibration_at is not None
            and episode >= self.pending_recalibration_at
        ):
            min_episode_id = self.recalibration_reference_min_episode
            # Do not replace a valid frozen witness until the replay contains
            # at least one complete post-trigger sample.  Otherwise snapshot()
            # would clear the pending recalibration marker and strand the
            # detector in a permanently-not-ready state.
            if self._batch(buffer, 1, min_episode_id=min_episode_id) is None:
                return {
                    "phase": "reference_recalibration",
                    "batches": int(self.n_batches_trained),
                    "residual": None,
                    "z": 0.0,
                }
            self.snapshot(episode)
            calibration = self.calibrate_reference(
                buffer,
                n_batches=max(20, int(self.window)),
                min_episode_id=min_episode_id,
            )
            self.recalibration_reference_min_episode = None
            return {
                "phase": "recalibrated",
                "batches": int(self.n_batches_trained),
                "residual": None,
                "z": 0.0,
                **calibration,
            }

        if not self.is_monitoring_ready():
            return {
                "phase": "reference_calibration",
                "batches": int(self.n_batches_trained),
                "residual": None,
                "z": 0.0,
            }

        res = self.measure(buffer)
        z = self.residual_z_score()

        return {
            "phase": "monitoring",
            "batches": int(self.n_batches_trained),
            "residual": res,
            "z": float(z),
            "standardized_residual": float(self.latest_standardized_residual),
            "cusum": float(self.cusum_stat),
            "cusum_threshold": float(self.cusum_threshold),
        }

    def notify_trigger(self, episode: int):
        """
        Schedule a new snapshot after the trigger starts adaptation. Without
        this step, an old witness would alarm forever after a real shift.
        """
        self.pending_recalibration_at = int(episode) + self.recalibrate_after
        self.recalibration_reference_min_episode = int(episode) + 1

    def get_diagnostics(self) -> Dict:
        return {
            "n_snapshots": int(self.n_snapshots),
            "n_batches_trained": int(self.n_batches_trained),
            "last_snapshot_episode": self.last_snapshot_episode,
            "pending_recalibration_at": self.pending_recalibration_at,
            "recalibration_reference_min_episode": self.recalibration_reference_min_episode,
            "latest_residual": (
                float(self.residual_history[-1])
                if self.residual_history else None
            ),
            "latest_standardized_residual": float(
                self.latest_standardized_residual
            ),
            "latest_z": float(self.latest_standardized_residual),
            "latest_cusum": float(self.cusum_stat),
            "frozen_ready": bool(self.frozen is not None),
            "monitoring_ready": self.is_monitoring_ready(),
            "reference_mean": self.reference_mean,
            "reference_std": self.reference_std,
            "reference_sample_count": int(self.reference_sample_count),
        }


class MatrixDriftDetector:
    """
    Diagnostic tracker for movement in the learned influence matrix.

    Finite-sample matrix movement can arise from estimator noise, support
    changes, policy drift, or a structural shift. It helps characterize an
    event but is not sufficient causal evidence and is not part of the final
    scheduler trigger.
    """

    def __init__(self, window: int = 20, eps: float = 1e-8):
        self.window = int(window)
        self.eps = float(eps)

        self.prev: Optional[np.ndarray] = None
        self.history: List[float] = []

    def update(self, W: np.ndarray) -> float:
        """Update from signed W[n_agents,n_agents] and return relative change."""
        W = np.asarray(W, dtype=np.float64)

        if self.prev is None:
            self.prev = W.copy()
            return 0.0

        num = float(np.linalg.norm(W - self.prev, ord="fro"))
        den = float(np.linalg.norm(self.prev, ord="fro")) + self.eps

        self.prev = W.copy()

        val = num / den
        self.history.append(val)

        if len(self.history) > 500:
            del self.history[:-500]

        return val

    def z_score(self) -> float:
        if len(self.history) < 5:
            return 0.0

        recent = np.asarray(self.history[-self.window:], dtype=np.float64)
        base = recent[:-1]

        if base.size < 3:
            return 0.0

        mu, sd = float(np.mean(base)), float(np.std(base))

        if sd < 1e-9:
            return 0.0

        return float((self.history[-1] - mu) / sd)

    def get_diagnostics(self) -> Dict:
        return {
            "latest_change": (
                float(self.history[-1]) if self.history else None
            ),
            "latest_z": float(self.z_score()),
            "n_observations": int(len(self.history)),
        }
