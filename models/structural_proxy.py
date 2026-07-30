import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCounterfactualProxyNet(nn.Module):
    """
    Một proxy network trong ensemble.

    Input theo đúng conditioning set của CIG-AMF:

        obs_i
        action_i
        action_j
        z_core_excl_j
        m_periph_excl_j
        belief_summary

    Output:
        predicted finite-horizon return R_i^(H)

    Ý nghĩa:
    - Đây không phải causal oracle.
    - Đây là supervised surrogate để học local counterfactual return model.
    - Structural score được lấy bằng cách so prediction dưới observed action
      với prediction dưới các perturbed alternative actions của neighbour j.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        core_dim,
        periph_dim,
        belief_dim,
        hidden=160,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.core_dim = int(core_dim)
        self.periph_dim = int(periph_dim)
        self.belief_dim = int(belief_dim)
        self.hidden = int(hidden)

        self.in_dim = (
            self.obs_dim
            + self.action_dim
            + self.action_dim
            + self.core_dim
            + self.periph_dim
            + self.belief_dim
        )

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(
        self,
        obs_i,
        action_i_onehot,
        action_j_onehot,
        z_core_excl_j,
        m_periph_excl_j,
        belief_summary,
    ):
        x = torch.cat(
            [
                obs_i,
                action_i_onehot,
                action_j_onehot,
                z_core_excl_j,
                m_periph_excl_j,
                belief_summary,
            ],
            dim=-1,
        )

        return self.net(x).squeeze(-1)


class LocalCounterfactualProxyEnsemble:
    """
    Ensemble local counterfactual proxy.

    Bám method:

        f_ij(s_i, a_i, a_j, Z_i^{-j}, M_i^{-j}, B_i) -> R_i^(H)

    Dùng để:
    - train supervised từ replay trajectory.
    - score local counterfactual effect cho directed pair (i, j).
    - trả mean effect và uncertainty magnitude.

    Sửa critical:
    - Không dùng "max absolute alternative effect" nữa.
    - score_batch() dùng mean absolute intervention effect across all
      alternative actions a'_j != a_j.
    - Điều này khớp hơn với tiny oracle, vì tiny oracle cũng lấy mean absolute
      effect qua candidate interventions.

    Sigma:
    - sigma trả ra là standard deviation across ensemble effects:
          sqrt(var + eps)
      không phải raw variance.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        core_dim,
        periph_dim,
        belief_dim,
        n_ensemble=3,
        hidden=160,
        lr=1e-3,
        buffer_size=200000,
        device="cpu",
        grad_clip=1.0,
        eps=1e-8,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.core_dim = int(core_dim)
        self.periph_dim = int(periph_dim)
        self.belief_dim = int(belief_dim)
        self.n_ensemble = int(n_ensemble)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.buffer_size = int(buffer_size)
        self.device = device
        self.grad_clip = float(grad_clip)
        self.eps = float(eps)

        self.models = [
            LocalCounterfactualProxyNet(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                core_dim=self.core_dim,
                periph_dim=self.periph_dim,
                belief_dim=self.belief_dim,
                hidden=self.hidden,
            ).to(self.device)
            for _ in range(self.n_ensemble)
        ]

        self.optims = [
            torch.optim.Adam(model.parameters(), lr=self.lr)
            for model in self.models
        ]

        self.buffer = deque(maxlen=self.buffer_size)

        self.last_train_called = False
        self.last_train_batch_count = 0
        self.latest_residual = 0.0
        self.latest_train_residual = 0.0
        self.latest_holdout_residual = 0.0
        self.latest_loss = 0.0

    # ============================================================
    # Tensor helpers
    # ============================================================

    def _one_hot(self, actions):
        if isinstance(actions, torch.Tensor):
            a = actions.to(device=self.device, dtype=torch.long)
        else:
            a = torch.tensor(actions, dtype=torch.long, device=self.device)

        if a.dim() == 0:
            a = a.unsqueeze(0)

        a = a.clamp(min=0, max=self.action_dim - 1)

        return F.one_hot(a, num_classes=self.action_dim).to(dtype=torch.float32)

    def _to_float_tensor(self, x, expected_dim=None):
        if isinstance(x, torch.Tensor):
            t = x.to(device=self.device, dtype=torch.float32)
        else:
            t = torch.tensor(
                np.asarray(x, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )

        if t.dim() == 1:
            t = t.unsqueeze(0)

        if expected_dim is not None and t.shape[-1] != int(expected_dim):
            raise ValueError(
                f"Expected last dim={expected_dim}, got {t.shape[-1]}"
            )

        return t

    def _normalise_vector(self, x, expected_dim):
        arr = np.asarray(x, dtype=np.float32).reshape(-1)

        if arr.shape[0] != int(expected_dim):
            raise ValueError(
                f"Expected vector dim={expected_dim}, got {arr.shape[0]}"
            )

        return arr.astype(np.float32)

    # ============================================================
    # Buffer API
    # ============================================================

    def add_sample(
        self,
        ego_id,
        neighbor_id,
        obs_i,
        action_i,
        observed_action_j,
        z_core_excl_j,
        m_periph_excl_j,
        belief_summary,
        target_return_h,
    ):
        """
        Add một supervised sample cho proxy buffer.
        """
        sample = {
            "ego_id": int(ego_id),
            "neighbor_id": int(neighbor_id),
            "obs_i": self._normalise_vector(obs_i, self.obs_dim),
            "action_i": int(action_i),
            "observed_action_j": int(observed_action_j),
            "z_core_excl_j": self._normalise_vector(
                z_core_excl_j,
                self.core_dim,
            ),
            "m_periph_excl_j": self._normalise_vector(
                m_periph_excl_j,
                self.periph_dim,
            ),
            "belief_summary": self._normalise_vector(
                belief_summary,
                self.belief_dim,
            ),
            "target_return_h": float(target_return_h),
        }

        self.buffer.append(sample)

    def get_buffer_size(self):
        return int(len(self.buffer))

    def get_last_train_called(self):
        return bool(self.last_train_called)

    def get_last_train_batch_count(self):
        return int(self.last_train_batch_count)

    def get_latest_residual(self):
        return float(self.latest_residual)

    def get_latest_train_residual(self):
        return float(self.latest_train_residual)

    def get_latest_holdout_residual(self):
        return float(self.latest_holdout_residual)

    def _sample_batch(self, batch_size):
        batch_size = int(batch_size)

        if len(self.buffer) == 0:
            return []

        n = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def _batch_to_tensors(self, batch):
        obs = np.stack(
            [b["obs_i"] for b in batch],
            axis=0,
        ).astype(np.float32)

        action_i = np.asarray(
            [b["action_i"] for b in batch],
            dtype=np.int64,
        )

        action_j = np.asarray(
            [b["observed_action_j"] for b in batch],
            dtype=np.int64,
        )

        z = np.stack(
            [b["z_core_excl_j"] for b in batch],
            axis=0,
        ).astype(np.float32)

        m = np.stack(
            [b["m_periph_excl_j"] for b in batch],
            axis=0,
        ).astype(np.float32)

        belief = np.stack(
            [b["belief_summary"] for b in batch],
            axis=0,
        ).astype(np.float32)

        target = np.asarray(
            [b["target_return_h"] for b in batch],
            dtype=np.float32,
        )

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_i_oh = self._one_hot(action_i)
        action_j_oh = self._one_hot(action_j)
        z_t = torch.tensor(z, dtype=torch.float32, device=self.device)
        m_t = torch.tensor(m, dtype=torch.float32, device=self.device)
        belief_t = torch.tensor(belief, dtype=torch.float32, device=self.device)
        target_t = torch.tensor(target, dtype=torch.float32, device=self.device)

        return obs_t, action_i_oh, action_j_oh, z_t, m_t, belief_t, target_t

    # ============================================================
    # Training
    # ============================================================

    def train_step(
        self,
        n_steps=1,
        batch_size=256,
        holdout_size=0,
    ):
        """
        Train ensemble bằng supervised regression.

        Nếu holdout_size > 0 và buffer đủ lớn, latest_residual được tính
        trên holdout batch không dùng để update gradient trong step đó.
        Nếu không đủ dữ liệu holdout, hàm tự fallback về train-batch residual
        để giữ backward compatibility với runner cũ.
        """
        self.last_train_called = True
        self.last_train_batch_count = 0

        if len(self.buffer) == 0:
            self.latest_loss = 0.0
            self.latest_residual = 0.0
            self.latest_train_residual = 0.0
            self.latest_holdout_residual = 0.0
            return 0.0

        n_steps = int(max(0, n_steps))
        batch_size = int(max(1, batch_size))
        holdout_size = int(max(0, holdout_size))

        if n_steps == 0:
            self.latest_loss = 0.0
            return 0.0

        all_losses = []
        latest_abs_residual = None
        latest_train_residual = None
        latest_holdout_residual = None

        for _ in range(n_steps):
            total_requested = batch_size + holdout_size
            n_total = min(len(self.buffer), total_requested)
            batch = random.sample(list(self.buffer), n_total)

            if len(batch) == 0:
                continue

            if holdout_size > 0 and len(batch) > holdout_size:
                holdout_batch = batch[:holdout_size]
                train_batch = batch[holdout_size:]
            else:
                holdout_batch = []
                train_batch = batch

            if len(train_batch) == 0:
                train_batch = batch
                holdout_batch = []

            (
                obs_t,
                action_i_oh,
                action_j_oh,
                z_t,
                m_t,
                belief_t,
                target_t,
            ) = self._batch_to_tensors(train_batch)

            preds_for_residual = []

            for model, optim in zip(self.models, self.optims):
                model.train()

                pred = model(
                    obs_i=obs_t,
                    action_i_onehot=action_i_oh,
                    action_j_onehot=action_j_oh,
                    z_core_excl_j=z_t,
                    m_periph_excl_j=m_t,
                    belief_summary=belief_t,
                )

                loss = F.mse_loss(pred, target_t)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.grad_clip,
                )
                optim.step()

                all_losses.append(float(loss.detach().cpu().item()))
                preds_for_residual.append(pred.detach())

            if len(preds_for_residual) > 0:
                pred_mean = torch.stack(preds_for_residual, dim=0).mean(dim=0)
                latest_train_residual = torch.mean(
                    torch.abs(pred_mean - target_t)
                ).detach()

            if len(holdout_batch) > 0:
                (
                    ho_obs_t,
                    ho_action_i_oh,
                    ho_action_j_oh,
                    ho_z_t,
                    ho_m_t,
                    ho_belief_t,
                    ho_target_t,
                ) = self._batch_to_tensors(holdout_batch)

                holdout_preds = []

                with torch.no_grad():
                    for model in self.models:
                        model.eval()
                        holdout_preds.append(
                            model(
                                obs_i=ho_obs_t,
                                action_i_onehot=ho_action_i_oh,
                                action_j_onehot=ho_action_j_oh,
                                z_core_excl_j=ho_z_t,
                                m_periph_excl_j=ho_m_t,
                                belief_summary=ho_belief_t,
                            )
                        )

                if len(holdout_preds) > 0:
                    holdout_pred_mean = torch.stack(
                        holdout_preds,
                        dim=0,
                    ).mean(dim=0)
                    latest_holdout_residual = torch.mean(
                        torch.abs(holdout_pred_mean - ho_target_t)
                    ).detach()
                    latest_abs_residual = latest_holdout_residual
            else:
                latest_holdout_residual = latest_train_residual
                latest_abs_residual = latest_train_residual

            self.last_train_batch_count += 1

        if len(all_losses) == 0:
            self.latest_loss = 0.0
            self.latest_residual = 0.0
            self.latest_train_residual = 0.0
            self.latest_holdout_residual = 0.0
            return 0.0

        self.latest_loss = float(np.mean(all_losses))

        if latest_train_residual is not None:
            self.latest_train_residual = float(
                latest_train_residual.cpu().item()
            )

        if latest_holdout_residual is not None:
            self.latest_holdout_residual = float(
                latest_holdout_residual.cpu().item()
            )

        if latest_abs_residual is not None:
            self.latest_residual = float(latest_abs_residual.cpu().item())

        return float(self.latest_loss)

    # ============================================================
    # Prediction / scoring
    # ============================================================

    def _predict_ensemble(
        self,
        obs_i,
        action_i,
        action_j,
        z_core_excl_j,
        m_periph_excl_j,
        belief_summary,
    ):
        """
        Predict return với từng ensemble member.

        Return:
            tensor shape [E, B]
        """
        obs_t = self._to_float_tensor(obs_i, expected_dim=self.obs_dim)
        z_t = self._to_float_tensor(z_core_excl_j, expected_dim=self.core_dim)
        m_t = self._to_float_tensor(m_periph_excl_j, expected_dim=self.periph_dim)
        belief_t = self._to_float_tensor(
            belief_summary,
            expected_dim=self.belief_dim,
        )

        if isinstance(action_i, torch.Tensor):
            action_i_arr = action_i.detach().cpu().numpy()
        else:
            action_i_arr = np.asarray(action_i)

        if isinstance(action_j, torch.Tensor):
            action_j_arr = action_j.detach().cpu().numpy()
        else:
            action_j_arr = np.asarray(action_j)

        action_i_arr = np.asarray(action_i_arr, dtype=np.int64).reshape(-1)
        action_j_arr = np.asarray(action_j_arr, dtype=np.int64).reshape(-1)

        action_i_oh = self._one_hot(action_i_arr)
        action_j_oh = self._one_hot(action_j_arr)

        preds = []

        with torch.no_grad():
            for model in self.models:
                model.eval()

                pred = model(
                    obs_i=obs_t,
                    action_i_onehot=action_i_oh,
                    action_j_onehot=action_j_oh,
                    z_core_excl_j=z_t,
                    m_periph_excl_j=m_t,
                    belief_summary=belief_t,
                )

                preds.append(pred)

        return torch.stack(preds, dim=0)

    def _counterfactual_effects_mean_abs(
        self,
        obs,
        action_i,
        action_j,
        z,
        m,
        belief,
    ):
        """
        Compute ensemble-level mean absolute intervention effect.

        For each sample b and each ensemble member e:

            effect[e, b] =
                mean_{a'_j != a_j} abs(
                    f_e(s_i, a_i, a'_j, Z^{-j}, M^{-j}, B_i)
                    -
                    f_e(s_i, a_i, a_j,   Z^{-j}, M^{-j}, B_i)
                )

        Đây là sửa quan trọng so với bản max-effect:
        - không lấy alternative gây effect cực đại.
        - không phóng đại influence bằng best/worst perturbation.
        - khớp hơn với tiny oracle lấy mean absolute effect.
        """
        batch_size = int(obs.shape[0])

        base_preds = self._predict_ensemble(
            obs_i=obs,
            action_i=action_i,
            action_j=action_j,
            z_core_excl_j=z,
            m_periph_excl_j=m,
            belief_summary=belief,
        )

        abs_effect_sum = torch.zeros_like(base_preds)
        valid_count = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=self.device,
        )

        for alt_action in range(self.action_dim):
            alt_arr = np.full(
                shape=(batch_size,),
                fill_value=int(alt_action),
                dtype=np.int64,
            )

            valid_mask_np = alt_arr != action_j

            if not np.any(valid_mask_np):
                continue

            alt_preds = self._predict_ensemble(
                obs_i=obs,
                action_i=action_i,
                action_j=alt_arr,
                z_core_excl_j=z,
                m_periph_excl_j=m,
                belief_summary=belief,
            )

            abs_effect = torch.abs(alt_preds - base_preds)

            valid_mask = torch.tensor(
                valid_mask_np,
                dtype=torch.float32,
                device=self.device,
            )

            abs_effect_sum = abs_effect_sum + abs_effect * valid_mask[None, :]
            valid_count = valid_count + valid_mask

        valid_count = torch.clamp(valid_count, min=1.0)
        effects = abs_effect_sum / valid_count[None, :]

        return effects

    def score_batch(
        self,
        obs_i_batch,
        action_i_batch,
        observed_action_j_batch,
        z_core_excl_j_batch,
        m_periph_excl_j_batch,
        belief_summary_batch,
    ):
        """
        Score directed pairs bằng standardized local counterfactual comparison.

        Discrete action perturbation:
            observed:    a_j
            alternatives: all a'_j != a_j

        Score:
            mu    = mean over ensemble of mean absolute alternative effect
            sigma = standard deviation over ensemble effects

        Return:
            mu_arr: np.ndarray shape [B]
            sigma_arr: np.ndarray shape [B]
        """
        if len(obs_i_batch) == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        obs = np.asarray(obs_i_batch, dtype=np.float32)
        z = np.asarray(z_core_excl_j_batch, dtype=np.float32)
        m = np.asarray(m_periph_excl_j_batch, dtype=np.float32)
        belief = np.asarray(belief_summary_batch, dtype=np.float32)

        action_i = np.asarray(action_i_batch, dtype=np.int64).reshape(-1)
        action_j = np.asarray(observed_action_j_batch, dtype=np.int64).reshape(-1)

        effects = self._counterfactual_effects_mean_abs(
            obs=obs,
            action_i=action_i,
            action_j=action_j,
            z=z,
            m=m,
            belief=belief,
        )

        mu = torch.mean(effects, dim=0)

        if effects.shape[0] <= 1:
            sigma = torch.zeros_like(mu)
        else:
            sigma = torch.sqrt(
                torch.var(effects, dim=0, unbiased=True) + self.eps
            )

        return (
            mu.detach().cpu().numpy().astype(np.float32),
            sigma.detach().cpu().numpy().astype(np.float32),
        )

    def score_pair(
        self,
        obs_i,
        action_i,
        observed_action_j,
        z_core_excl_j,
        m_periph_excl_j,
        belief_summary,
    ):
        """
        Convenience wrapper cho tiny oracle calibration và debug.

        Return:
            (mu, sigma)
        """
        mu_arr, sigma_arr = self.score_batch(
            obs_i_batch=[obs_i],
            action_i_batch=[int(action_i)],
            observed_action_j_batch=[int(observed_action_j)],
            z_core_excl_j_batch=[z_core_excl_j],
            m_periph_excl_j_batch=[m_periph_excl_j],
            belief_summary_batch=[belief_summary],
        )

        if len(mu_arr) == 0:
            return 0.0, 0.0

        return float(mu_arr[0]), float(sigma_arr[0])