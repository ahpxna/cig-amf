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
    need not predict exceptionally well; its error only needs to change when
    the environment law changes. obs_dim/action_dim match the main proxy,
    hidden should remain small (64) because larger models overfit and make the
    residual noisy, and n_horizons matches the main proxy.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 64,
        n_horizons: int = 3,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_horizons = int(n_horizons)

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
        cusum_allowance: float = 0.5,
        cusum_threshold: float = 8.0,
        seed: int = 0,
    ):
        torch.manual_seed(int(seed))

        self.device = device
        self.action_dim = int(action_dim)
        self.n_horizons = int(n_horizons)
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
        ).to(device)

        self.optim = torch.optim.Adam(self.live.parameters(), lr=float(lr))

        # Frozen baseline. None until warm-up is complete.
        self.frozen: Optional[StructuralDriftProbe] = None

        self.rng = np.random.RandomState(int(seed))

        self.n_batches_trained = 0
        self.residual_history: List[float] = []
        self.last_snapshot_episode = None
        self.pending_recalibration_at = None
        self.n_snapshots = 0

    # ------------------------------------------------------------------

    def _one_hot(self, a: np.ndarray) -> torch.Tensor:
        t = torch.tensor(
            np.asarray(a, dtype=np.int64), dtype=torch.long, device=self.device
        ).clamp(0, self.action_dim - 1)

        return F.one_hot(t, num_classes=self.action_dim).to(dtype=torch.float32)

    def _batch(self, buffer, n: int):
        """Get batch from proxy replay. Return tensors or None."""
        if buffer is None or len(buffer) == 0:
            return None

        buf = list(buffer)
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
            np.stack(
                [b.get("target_lag_rewards", b.get("target_returns_multi")) for b in batch],
                axis=0,
            ),
            dtype=torch.float32, device=self.device,
        )                                                    # [B, H]

        return obs, a_i, a_j, tgt

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
        self.cusum_stat = 0.0
        self.latest_standardized_residual = 0.0

    def measure(self, buffer, n: int = 512) -> Optional[float]:
        """
        Evaluate the frozen model on the newest data.

        Draw from the end of the buffer rather than randomly because the
        question is whether the world changed recently. Return mean absolute
        error at the final horizon, or None.
        """
        if self.frozen is None or buffer is None or len(buffer) == 0:
            return None

        buf = list(buffer)[-int(n):]

        obs = torch.tensor(
            np.stack([b["obs_i"] for b in buf], axis=0),
            dtype=torch.float32, device=self.device,
        )
        a_i = self._one_hot([b["action_i"] for b in buf])
        a_j = self._one_hot([b["observed_action_j"] for b in buf])
        tgt = torch.tensor(
            np.stack(
                [b.get("target_lag_rewards", b.get("target_returns_multi")) for b in buf],
                axis=0,
            ),
            dtype=torch.float32, device=self.device,
        )

        with torch.no_grad():
            pred = self.frozen(obs, a_i, a_j)                  # [B, H]
            res = torch.mean(torch.abs(pred[:, -1] - tgt[:, -1]))

        val = float(res.cpu().item())
        self.residual_history.append(val)

        if len(self.residual_history) > 500:
            del self.residual_history[:-500]

        return val

    def residual_z_score(self) -> float:
        """
        Standardize against a fixed pre-monitoring reference distribution.
        """
        h = self.residual_history

        if self.reference_mean is None:
            if len(h) < max(5, self.window):
                return 0.0
            reference = np.asarray(h[:self.window], dtype=np.float64)
            self.reference_mean = float(np.mean(reference))
            self.reference_std = float(max(np.std(reference), 1e-9))
            return 0.0
        z = float(
            (h[-1] - self.reference_mean) / self.reference_std
        )
        self.latest_standardized_residual = z
        self.cusum_stat = max(
            0.0, self.cusum_stat + z - self.cusum_allowance
        )
        return float(self.cusum_stat)

    # ------------------------------------------------------------------

    def step(self, episode: int, buffer, n_train_batches: int = 5) -> Dict:
        """Run the complete lifecycle once per episode and return log state."""
        # Train normally before the first snapshot.
        if self.frozen is None:
            self.train_batches(buffer, n_train_batches)

            if self.n_batches_trained >= self.warmup_batches:
                self.snapshot(episode)

            return {
                "phase": "warmup",
                "batches": int(self.n_batches_trained),
                "residual": None,
                "z": 0.0,
            }

        # After freezing, the live model keeps training for the next snapshot.
        # The frozen witness is never updated.
        self.train_batches(buffer, n_train_batches)

        res = self.measure(buffer)
        z = self.residual_z_score()

        # Has it been time for a new photo yet?
        if (
            self.pending_recalibration_at is not None
            and episode >= self.pending_recalibration_at
        ):
            self.snapshot(episode)

            return {
                "phase": "recalibrated",
                "batches": int(self.n_batches_trained),
                "residual": res,
                "z": 0.0,
            }

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

    def get_diagnostics(self) -> Dict:
        return {
            "n_snapshots": int(self.n_snapshots),
            "n_batches_trained": int(self.n_batches_trained),
            "last_snapshot_episode": self.last_snapshot_episode,
            "pending_recalibration_at": self.pending_recalibration_at,
            "latest_residual": (
                float(self.residual_history[-1])
                if self.residual_history else None
            ),
            "latest_z": float(self.residual_z_score()),
            "frozen_ready": bool(self.frozen is not None),
        }


class MatrixDriftDetector:
    """
    Independent second trigger that tracks the influence matrix.

    THEORY (Pieroth, ICML 2024, Theorem 5.11): influence measures are
    continuous in policy parameters. Gradual behavioural policy drift can
    therefore change the influence matrix only gradually. A discontinuous
    matrix jump cannot be behavioural drift and must indicate structural
    change. This is a mathematical basis for the two-tier separation rather
    than an intuition.

    Compared with the residual probe, the matrix trigger need not wait H steps
    for return and can respond earlier. Its disadvantage is that the matrix
    comes from a proxy trained on H-step returns, so delay remains indirectly.
    Use both: the probe detects changes in environment law, while the matrix
    detects changes in influence structure; these events need not coincide.
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
