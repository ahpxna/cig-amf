"""
structural_proxy_v2.py — Local counterfactual proxy, bản vá.

=============================================================================
BỐN LỖI CỦA BẢN v1 ĐƯỢC SỬA Ở ĐÂY
=============================================================================

[L1] DẤU BỊ HUỶ NGAY TRONG ESTIMATOR.
     v1: abs_effect = torch.abs(alt_preds - base_preds)  -> mu LUÔN >= 0.
     Hệ quả dây chuyền:
       - p_core buộc phải dùng |mu_bar| (vì mu đã không âm sẵn)
       - không thể phân biệt "thằng ngáng đường" với "thằng hỗ trợ"
       - KHÔNG THỂ khớp với oracle của env (oracle CÓ DẤU) -> Exp3 bất khả thi
     v2: giữ dấu. Expose 4 chế độ (xem effect_mode) để chạy ablation.

[L2] ƯỚC LƯỢNG PLUG-IN, THỪA HƯỞNG TOÀN BỘ BIAS CỦA REWARD MODEL.
     v1: w = f(a') - f(a). Nếu f lệch thì w lệch. Mà a_j không ngẫu nhiên
     nên f học được từ dữ liệu confounded.
     v2: doubly-robust. Vì trong MARL ta BIẾT CHÍNH XÁC pi_j (ta tự train nó),
     propensity là exact -> DR không chệch ngay cả khi f sai.
     Đây là lợi thế mà nhà thống kê y học mơ không được.

[L3] ENSEMBLE GIẢ — cả 3 member train trên CÙNG batch, cùng thứ tự.
     v1: `for model, optim in zip(self.models, self.optims)` nằm TRONG vòng
     lặp batch -> mọi member thấy đúng cùng dữ liệu -> hội tụ về gần cùng
     một hàm -> sigma = 0.000 (đúng như bảng kết quả trong paper).
     v2: mỗi member có bootstrap mask riêng + batch riêng + khởi tạo riêng.

[L4] CHỈ MỘT HORIZON -> không có chiều "độ trễ" cho influence signature.
     v2: multi-horizon head, dự đoán R^(1), R^(2), ..., R^(H) cùng lúc.
     Rẻ (chỉ đổi output layer) mà cho ngay chiều latency.

=============================================================================
BACKWARD COMPATIBILITY
=============================================================================
Giữ nguyên chữ ký các hàm runner đang gọi:
    add_sample(...)  -> thêm 2 tham số optional, có default
    train_step(...)  -> giữ nguyên
    score_batch(...) -> giữ nguyên, VẪN trả (mu_arr, sigma_arr)
    score_pair(...)  -> giữ nguyên
Thêm mới:
    score_batch_full(...) -> trả dict đầy đủ để xây influence signature
=============================================================================
"""

import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Mạng proxy
# =============================================================================

class LocalCounterfactualProxyNet(nn.Module):
    """
    Một member trong ensemble.

    Input (đúng conditioning set của paper, Eq. 5):
        obs_i, a_i, a_j, Z_i^{-j}, M_i^{-j}, B_i

    Output:
        [B, n_horizons] — dự đoán R_i^(1), R_i^(2), ..., R_i^(H)

    Vì sao multi-horizon:
        Ảnh hưởng của một neighbour có ĐỘ TRỄ. Blocker chặn đường tác động
        tức thì (h=1). Relay/signaller phát tín hiệu thì lợi ích chỉ hiện ra
        sau vài bước (h=3). Nếu chỉ dự đoán R^(H) tổng, hai loại này trông
        giống hệt nhau. Tách theo horizon là chiều thứ 6 của signature.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        core_dim: int,
        periph_dim: int,
        belief_dim: int,
        hidden: int = 160,
        n_horizons: int = 3,
        use_belief_input: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.core_dim = int(core_dim)
        self.periph_dim = int(periph_dim)
        self.belief_dim = int(belief_dim)
        self.hidden = int(hidden)
        self.n_horizons = int(n_horizons)

        # ---------------------------------------------------------------
        # [H6] CẮT VÒNG LẶP PHẢN HỒI belief -> proxy -> belief
        #
        # Bản v1 đưa B_i vào input của proxy. Nhưng B_i được SINH RA TỪ
        # chính w_hat mà proxy này tạo ra. Hệ tự xác nhận: nếu B_i nói
        # "j là core" thì proxy có thể học dự đoán hiệu ứng lớn hơn cho
        # core member -> p_core tăng thêm -> belief càng chắc chắn.
        # Đó là confounder do chính kiến trúc tạo ra.
        #
        # Mặc định v2 TẮT belief input. Bật lại chỉ khi muốn chạy ablation.
        # ---------------------------------------------------------------
        self.use_belief_input = bool(use_belief_input)

        self.in_dim = (
            self.obs_dim
            + self.action_dim   # a_i one-hot
            + self.action_dim   # a_j one-hot
            + self.core_dim
            + self.periph_dim
            + (self.belief_dim if self.use_belief_input else 0)
        )

        layers = [
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))

        layers += [
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))

        # Output n_horizons thay vì 1.
        layers.append(nn.Linear(self.hidden, self.n_horizons))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        obs_i: torch.Tensor,            # [B, obs_dim]
        action_i_onehot: torch.Tensor,  # [B, action_dim]
        action_j_onehot: torch.Tensor,  # [B, action_dim]
        z_core_excl_j: torch.Tensor,    # [B, core_dim]
        m_periph_excl_j: torch.Tensor,  # [B, periph_dim]
        belief_summary: torch.Tensor,   # [B, belief_dim]
    ) -> torch.Tensor:
        """
        Returns:
            [B, n_horizons]
        """
        parts = [
            obs_i,
            action_i_onehot,
            action_j_onehot,
            z_core_excl_j,
            m_periph_excl_j,
        ]

        if self.use_belief_input:
            parts.append(belief_summary)

        x = torch.cat(parts, dim=-1)  # [B, in_dim]

        return self.net(x)  # [B, n_horizons]


# =============================================================================
# Ensemble
# =============================================================================

class LocalCounterfactualProxyEnsemble:
    """
    Ensemble proxy có dấu, doubly-robust, đa horizon.

    ---------------------------------------------------------------------
    BỐN CHẾ ĐỘ TÍNH EFFECT (effect_mode)
    ---------------------------------------------------------------------
    Cả bốn đều tính được từ cùng một forward pass, nên chạy ablation gần
    như miễn phí. Chọn mode nào tuỳ mục đích:

    "signed_aristocrat"  (MẶC ĐỊNH — dùng cho gán vai trò Thiện/Ác)
        w = f(s, a_j_obs) - E_{a' ~ pi_j}[ f(s, a') ]
        Đây là aristocrat utility (Wolpert & Tumer 2002) áp lên NEIGHBOUR
        thay vì lên chính mình như COMA (Foerster 2018).
        w > 0: hành động thực của j TỐT cho i hơn mức trung bình của j -> j đang GIÚP
        w < 0: j đang HẠI
        Baseline lấy kỳ vọng theo TOÀN BỘ pi_j nên phương sai thấp hơn hẳn
        so với việc bốc một hành động thay thế tuỳ tiện như v1.

    "signed_oracle_matched"  (dùng cho Exp3 calibration)
        w = mean_{a in candidates}[ f(s,a) ] - f(s, a_j_obs)
        Khớp ĐÚNG công thức oracle trong env. Chỉ mode này mới so được
        với compute_oracle_influence_from_current_state().

    "range"  (Pieroth ICML'24 style — baseline đối chứng)
        w = max_a f(s,a) - min_a f(s,a)   (luôn >= 0)
        Chính là impact sample U^{j->i} của Pieroth. Dùng làm BASELINE:
        nếu signed signature của ta phân vai tốt hơn range không dấu này
        thì đó là bằng chứng trực tiếp cho novelty.

    "mean_abs"  (bản v1 — giữ để chạy ablation "trước/sau khi vá")
        w = mean_{a != a_obs} |f(s,a) - f(s,a_obs)|

    ---------------------------------------------------------------------
    DOUBLY ROBUST — vì sao và công thức
    ---------------------------------------------------------------------
    Ước lượng plug-in cho E[R | do(a_j = a)] là f_hat(s,a). Nếu f_hat lệch,
    kết quả lệch. DR thêm một số hạng hiệu chỉnh dùng propensity:

        psi_DR(a) = f_hat(s,a) + (1{a_obs = a} / b_j(a|s)) * (R_obs - f_hat(s,a_obs))

    Với baseline aristocrat, tổng theo pi_j có dạng đóng rất gọn:

        w_DR = [f_hat(s,a_obs) - sum_a pi_j(a) f_hat(s,a)]        <- phần plug-in
             + (R_obs - f_hat(s,a_obs)) * (1/b_j(a_obs) - 1)       <- phần hiệu chỉnh

    Đọc bằng lời:
      - f_hat hoàn hảo -> residual = 0 -> số hạng hiệu chỉnh biến mất -> plug-in đúng.
      - f_hat lệch     -> residual != 0 -> hiệu chỉnh kéo về, với trọng số
                          tỉ lệ nghịch propensity.
    Chỉ cần MỘT trong hai (outcome model HOẶC propensity) đúng là không chệch.
    Trong MARL ta biết chính xác propensity -> luôn có sẵn một cái đúng.

    Importance weight 1/b bị CLIP để tránh nổ phương sai khi b nhỏ.
    """

    # Bốn chế độ hợp lệ
    MODES = (
        "signed_aristocrat",
        "signed_oracle_matched",
        "range",
        "mean_abs",
    )

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        core_dim: int,
        periph_dim: int,
        belief_dim: int,
        n_ensemble: int = 4,
        hidden: int = 160,
        lr: float = 1e-3,
        buffer_size: int = 200000,
        device: str = "cpu",
        grad_clip: float = 1.0,
        eps: float = 1e-8,
        # ---- mới ở v2 ----
        n_horizons: int = 3,
        effect_mode: str = "signed_aristocrat",
        use_doubly_robust: bool = True,
        iw_clip: float = 10.0,
        bootstrap_ratio: float = 0.8,
        use_belief_input: bool = False,
        candidate_actions: Optional[List[int]] = None,
        ensemble_dropout: float = 0.0,
        seed: int = 0,
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

        self.n_horizons = int(n_horizons)

        if effect_mode not in self.MODES:
            raise ValueError(
                f"effect_mode phải thuộc {self.MODES}, nhận '{effect_mode}'"
            )
        self.effect_mode = str(effect_mode)

        self.use_doubly_robust = bool(use_doubly_robust)
        self.iw_clip = float(iw_clip)
        self.bootstrap_ratio = float(np.clip(bootstrap_ratio, 0.1, 1.0))
        self.use_belief_input = bool(use_belief_input)

        self.candidate_actions = (
            list(range(self.action_dim))
            if candidate_actions is None
            else [int(a) for a in candidate_actions]
        )

        # ---------------------------------------------------------------
        # [L3] ENSEMBLE ĐA DẠNG THẬT
        # Ba nguồn đa dạng:
        #   1. Khởi tạo khác nhau (torch seed khác nhau mỗi member)
        #   2. Bootstrap mask khác nhau (mỗi member chỉ thấy 1 phần buffer)
        #   3. Batch khác nhau (sample riêng cho từng member, không dùng chung)
        # ---------------------------------------------------------------
        self.models = []

        for k in range(self.n_ensemble):
            torch.manual_seed(int(seed) * 1000 + k)

            self.models.append(
                LocalCounterfactualProxyNet(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    core_dim=self.core_dim,
                    periph_dim=self.periph_dim,
                    belief_dim=self.belief_dim,
                    hidden=self.hidden,
                    n_horizons=self.n_horizons,
                    use_belief_input=self.use_belief_input,
                    dropout=float(ensemble_dropout),
                ).to(self.device)
            )

        self.optims = [
            torch.optim.Adam(m.parameters(), lr=self.lr) for m in self.models
        ]

        # RNG riêng cho từng member để bootstrap mask độc lập.
        self._member_rngs = [
            random.Random(int(seed) * 7919 + k) for k in range(self.n_ensemble)
        ]

        self.buffer = deque(maxlen=self.buffer_size)

        # Diagnostics (runner cũ đang đọc các field này)
        self.last_train_called = False
        self.last_train_batch_count = 0
        self.latest_residual = 0.0
        self.latest_train_residual = 0.0
        self.latest_holdout_residual = 0.0
        self.latest_loss = 0.0

        # Diagnostics mới
        self.latest_ensemble_disagreement = 0.0
        self.latest_dr_correction_magnitude = 0.0
        self.n_interventional_samples = 0

    # =====================================================================
    # Helper tensor
    # =====================================================================

    def _one_hot(self, actions) -> torch.Tensor:
        """actions: array-like [B] -> [B, action_dim] float32"""
        if isinstance(actions, torch.Tensor):
            a = actions.to(device=self.device, dtype=torch.long)
        else:
            a = torch.tensor(
                np.asarray(actions, dtype=np.int64),
                dtype=torch.long,
                device=self.device,
            )

        if a.dim() == 0:
            a = a.unsqueeze(0)

        a = a.clamp(min=0, max=self.action_dim - 1)

        return F.one_hot(a, num_classes=self.action_dim).to(dtype=torch.float32)

    def _to_float_tensor(self, x, expected_dim=None) -> torch.Tensor:
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

    def _normalise_vector(self, x, expected_dim) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)

        if arr.shape[0] != int(expected_dim):
            raise ValueError(
                f"Expected vector dim={expected_dim}, got {arr.shape[0]}"
            )

        return arr.astype(np.float32)

    # =====================================================================
    # Buffer
    # =====================================================================

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
        # ---- mới ở v2, đều có default nên runner cũ vẫn chạy ----
        target_returns_multi=None,
        behaviour_prob_j=None,
        was_forced=False,
        state_key=None,
    ):
        """
        Thêm một mẫu supervised.

        Args mới:
            target_returns_multi:
                list/array length n_horizons = [R^(1), R^(2), ..., R^(H)].
                Nếu None, sẽ broadcast target_return_h ra mọi horizon
                (kém chính xác, chỉ để tương thích ngược — nên truyền vào).
            behaviour_prob_j:
                b_j(a_j_obs | s) tại thời điểm thu thập — CẦN cho DR.
                Nếu None, DR sẽ tự tắt cho mẫu này (fallback plug-in).
            was_forced:
                True nếu hành động của j bị eps-forcing ép. Mẫu này là
                CAN THIỆP THẬT, đáng giá hơn nhiều -> được oversample khi train.
            state_key:
                định danh ngữ cảnh (ví dụ zone id, hoặc hash vị trí thô).
                Dùng để tính chiều "context-conditionality" của signature.
        """
        if target_returns_multi is None:
            multi = np.full(
                (self.n_horizons,), float(target_return_h), dtype=np.float32
            )
        else:
            multi = np.asarray(target_returns_multi, dtype=np.float32).reshape(-1)

            if multi.shape[0] != self.n_horizons:
                raise ValueError(
                    f"target_returns_multi phải có length {self.n_horizons}, "
                    f"nhận {multi.shape[0]}"
                )

        sample = {
            "ego_id": int(ego_id),
            "neighbor_id": int(neighbor_id),
            "obs_i": self._normalise_vector(obs_i, self.obs_dim),
            "action_i": int(action_i),
            "observed_action_j": int(observed_action_j),
            "z_core_excl_j": self._normalise_vector(z_core_excl_j, self.core_dim),
            "m_periph_excl_j": self._normalise_vector(
                m_periph_excl_j, self.periph_dim
            ),
            "belief_summary": self._normalise_vector(
                belief_summary, self.belief_dim
            ),
            "target_return_h": float(target_return_h),
            "target_returns_multi": multi,                      # [n_horizons]
            "behaviour_prob_j": (
                None if behaviour_prob_j is None else float(behaviour_prob_j)
            ),
            "was_forced": bool(was_forced),
            "state_key": state_key,
        }

        self.buffer.append(sample)

        if bool(was_forced):
            self.n_interventional_samples += 1

    def get_buffer_size(self) -> int:
        return int(len(self.buffer))

    def get_last_train_called(self) -> bool:
        return bool(self.last_train_called)

    def get_last_train_batch_count(self) -> int:
        return int(self.last_train_batch_count)

    def get_latest_residual(self) -> float:
        return float(self.latest_residual)

    def get_latest_train_residual(self) -> float:
        return float(self.latest_train_residual)

    def get_latest_holdout_residual(self) -> float:
        return float(self.latest_holdout_residual)

    # =====================================================================
    # Train
    # =====================================================================

    def _batch_to_tensors(self, batch):
        """batch: list[dict] length B -> tuple tensors"""
        obs = np.stack([b["obs_i"] for b in batch], axis=0)          # [B, obs_dim]
        action_i = np.asarray([b["action_i"] for b in batch], np.int64)      # [B]
        action_j = np.asarray(
            [b["observed_action_j"] for b in batch], np.int64
        )                                                                     # [B]
        z = np.stack([b["z_core_excl_j"] for b in batch], axis=0)    # [B, core_dim]
        m = np.stack([b["m_periph_excl_j"] for b in batch], axis=0)  # [B, periph_dim]
        belief = np.stack(
            [b["belief_summary"] for b in batch], axis=0
        )                                                            # [B, belief_dim]
        target_multi = np.stack(
            [b["target_returns_multi"] for b in batch], axis=0
        )                                                            # [B, n_horizons]

        return (
            torch.tensor(obs, dtype=torch.float32, device=self.device),
            self._one_hot(action_i),
            self._one_hot(action_j),
            torch.tensor(z, dtype=torch.float32, device=self.device),
            torch.tensor(m, dtype=torch.float32, device=self.device),
            torch.tensor(belief, dtype=torch.float32, device=self.device),
            torch.tensor(target_multi, dtype=torch.float32, device=self.device),
        )

    def _sample_for_member(self, member_idx: int, n: int, forced_boost: float = 3.0):
        """
        [L3] Sample RIÊNG cho từng ensemble member.

        Hai cơ chế:
          - bootstrap: mỗi member chỉ được thấy `bootstrap_ratio` phần buffer
            (chọn theo hash ổn định của index để mask nhất quán qua các lần gọi)
          - oversample mẫu can thiệp: mẫu was_forced=True là can thiệp THẬT
            (đã cắt confounding) nên đáng giá hơn -> tăng xác suất được chọn.
        """
        if len(self.buffer) == 0:
            return []

        rng = self._member_rngs[member_idx]
        buf = list(self.buffer)

        # Bootstrap mask ổn định theo member: dùng hash của (member, vị trí).
        keep_prob = self.bootstrap_ratio
        weights = []
        pool = []

        for idx, s in enumerate(buf):
            # Hash ổn định -> cùng một member luôn thấy cùng một tập con.
            h = ((idx * 2654435761) ^ (member_idx * 40503)) & 0xFFFFFFFF
            if (h / 0xFFFFFFFF) > keep_prob:
                continue

            pool.append(s)
            weights.append(forced_boost if s["was_forced"] else 1.0)

        if len(pool) == 0:
            pool = buf
            weights = [
                forced_boost if s["was_forced"] else 1.0 for s in buf
            ]

        n = int(min(n, len(pool)))

        if n <= 0:
            return []

        # random.choices lấy có hoàn lại, có trọng số -> đúng ý oversample.
        return rng.choices(pool, weights=weights, k=n)

    def train_step(
        self,
        n_steps: int = 1,
        batch_size: int = 256,
        holdout_size: int = 0,
    ) -> float:
        """
        Train ensemble. Giữ nguyên chữ ký v1.

        Khác v1 ở chỗ: mỗi member sample batch RIÊNG (xem _sample_for_member).
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
        train_residuals = []

        for _ in range(n_steps):
            for k, (model, optim) in enumerate(zip(self.models, self.optims)):
                batch = self._sample_for_member(k, batch_size)

                if len(batch) == 0:
                    continue

                (
                    obs_t,
                    a_i_oh,
                    a_j_oh,
                    z_t,
                    m_t,
                    belief_t,
                    target_multi_t,   # [B, n_horizons]
                ) = self._batch_to_tensors(batch)

                model.train()

                pred = model(
                    obs_i=obs_t,
                    action_i_onehot=a_i_oh,
                    action_j_onehot=a_j_oh,
                    z_core_excl_j=z_t,
                    m_periph_excl_j=m_t,
                    belief_summary=belief_t,
                )  # [B, n_horizons]

                loss = F.mse_loss(pred, target_multi_t)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optim.step()

                all_losses.append(float(loss.detach().cpu().item()))

                # Residual đo trên horizon cuối (khớp R^(H) của v1).
                with torch.no_grad():
                    res = torch.mean(
                        torch.abs(pred[:, -1] - target_multi_t[:, -1])
                    )
                    train_residuals.append(float(res.cpu().item()))

            self.last_train_batch_count += 1

        # ---- holdout residual: batch chung, KHÔNG dùng để update ----------
        # [H7] Residual phải đo trên dữ liệu không tham gia gradient, nếu không
        # nó phản ánh chính sự thay đổi của mình chứ không phải structural shift.
        holdout_residual = None

        if holdout_size > 0 and len(self.buffer) > holdout_size:
            ho_batch = random.sample(list(self.buffer), int(holdout_size))

            (
                ho_obs,
                ho_ai,
                ho_aj,
                ho_z,
                ho_m,
                ho_b,
                ho_target,
            ) = self._batch_to_tensors(ho_batch)

            with torch.no_grad():
                preds = []
                for model in self.models:
                    model.eval()
                    preds.append(
                        model(
                            obs_i=ho_obs,
                            action_i_onehot=ho_ai,
                            action_j_onehot=ho_aj,
                            z_core_excl_j=ho_z,
                            m_periph_excl_j=ho_m,
                            belief_summary=ho_b,
                        )
                    )

                stacked = torch.stack(preds, dim=0)          # [E, B, n_horizons]
                pred_mean = stacked.mean(dim=0)              # [B, n_horizons]

                holdout_residual = float(
                    torch.mean(
                        torch.abs(pred_mean[:, -1] - ho_target[:, -1])
                    ).cpu().item()
                )

                # [L3] Chẩn đoán: ensemble có thật sự bất đồng không?
                # Nếu số này ~ 0 thì ensemble đang giả, sigma vô nghĩa.
                if stacked.shape[0] > 1:
                    self.latest_ensemble_disagreement = float(
                        torch.mean(torch.std(stacked, dim=0)).cpu().item()
                    )

        if len(all_losses) == 0:
            self.latest_loss = 0.0
            self.latest_residual = 0.0
            return 0.0

        self.latest_loss = float(np.mean(all_losses))
        self.latest_train_residual = (
            float(np.mean(train_residuals)) if train_residuals else 0.0
        )

        if holdout_residual is not None:
            self.latest_holdout_residual = holdout_residual
            self.latest_residual = holdout_residual
        else:
            self.latest_holdout_residual = self.latest_train_residual
            self.latest_residual = self.latest_train_residual

        return float(self.latest_loss)

    # =====================================================================
    # Dự đoán mọi hành động thay thế
    # =====================================================================

    def _predict_all_actions(
        self,
        obs,       # [B, obs_dim]
        action_i,  # [B]
        z,         # [B, core_dim]
        m,         # [B, periph_dim]
        belief,    # [B, belief_dim]
    ) -> torch.Tensor:
        """
        Dự đoán return cho MỌI hành động khả dĩ của j, với mọi ensemble member.

        Đây là tối ưu quan trọng: v1 gọi _predict_ensemble một lần cho mỗi
        alternative action (tức action_dim lần forward). v2 gộp thành một
        batch lớn -> chỉ 1 forward pass mỗi member.

        Returns:
            [E, B, A, n_horizons]
              E = số ensemble member
              B = batch
              A = action_dim
              n_horizons = số horizon
        """
        obs_t = self._to_float_tensor(obs, self.obs_dim)        # [B, obs_dim]
        z_t = self._to_float_tensor(z, self.core_dim)           # [B, core_dim]
        m_t = self._to_float_tensor(m, self.periph_dim)         # [B, periph_dim]
        belief_t = self._to_float_tensor(belief, self.belief_dim)  # [B, belief_dim]

        B = int(obs_t.shape[0])
        A = int(self.action_dim)

        a_i_oh = self._one_hot(np.asarray(action_i).reshape(-1))  # [B, A]

        # Nhân bản mỗi sample A lần, mỗi bản gán một hành động khác của j.
        # repeat_interleave: [B, D] -> [B*A, D] theo thứ tự
        #   sample0-act0, sample0-act1, ..., sample0-act(A-1), sample1-act0, ...
        obs_rep = obs_t.repeat_interleave(A, dim=0)        # [B*A, obs_dim]
        z_rep = z_t.repeat_interleave(A, dim=0)            # [B*A, core_dim]
        m_rep = m_t.repeat_interleave(A, dim=0)            # [B*A, periph_dim]
        belief_rep = belief_t.repeat_interleave(A, dim=0)  # [B*A, belief_dim]
        a_i_rep = a_i_oh.repeat_interleave(A, dim=0)       # [B*A, A]

        # a_j one-hot: lặp identity B lần -> [B*A, A]
        eye = torch.eye(A, dtype=torch.float32, device=self.device)  # [A, A]
        a_j_rep = eye.repeat(B, 1)                                    # [B*A, A]

        outs = []

        with torch.no_grad():
            for model in self.models:
                model.eval()

                pred = model(
                    obs_i=obs_rep,
                    action_i_onehot=a_i_rep,
                    action_j_onehot=a_j_rep,
                    z_core_excl_j=z_rep,
                    m_periph_excl_j=m_rep,
                    belief_summary=belief_rep,
                )  # [B*A, n_horizons]

                outs.append(pred.view(B, A, self.n_horizons))  # [B, A, n_horizons]

        return torch.stack(outs, dim=0)  # [E, B, A, n_horizons]

    # =====================================================================
    # Tính effect
    # =====================================================================

    def _compute_effects(
        self,
        preds_all: torch.Tensor,        # [E, B, A, n_horizons]
        action_j_obs: np.ndarray,       # [B]
        policy_probs_j: Optional[np.ndarray] = None,   # [B, A]
        observed_returns: Optional[np.ndarray] = None,  # [B]
        behaviour_probs_obs: Optional[np.ndarray] = None,  # [B]
        mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tính effect theo mode, có/không DR.

        Returns dict:
            effect:        [E, B]              effect ở horizon cuối
            effect_per_h:  [E, B, n_horizons]  effect tách theo từng horizon
            dr_correction: [B]                 độ lớn số hạng hiệu chỉnh DR
        """
        mode = self.effect_mode if mode is None else str(mode)

        E, B, A, H = preds_all.shape

        idx_obs = torch.tensor(
            np.asarray(action_j_obs, dtype=np.int64).reshape(-1),
            dtype=torch.long,
            device=self.device,
        ).clamp(0, A - 1)  # [B]

        # f(s, a_obs) cho mọi member, mọi horizon
        # gather cần index shape khớp -> mở rộng thành [E, B, 1, H]
        idx_exp = (
            idx_obs.view(1, B, 1, 1).expand(E, B, 1, H)
        )  # [E, B, 1, H]

        f_obs = torch.gather(preds_all, dim=2, index=idx_exp).squeeze(2)
        # f_obs: [E, B, H]

        dr_correction = torch.zeros(B, dtype=torch.float32, device=self.device)

        # ---------------------------------------------------------------
        if mode == "signed_aristocrat":
            # baseline = E_{a ~ pi_j}[ f(s,a) ]
            if policy_probs_j is None:
                # Không có pi_j -> dùng uniform (kém chính xác nhưng vẫn chạy)
                w = torch.full(
                    (B, A), 1.0 / float(A), dtype=torch.float32, device=self.device
                )
            else:
                w = torch.tensor(
                    np.asarray(policy_probs_j, dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )  # [B, A]

            # baseline: [E, B, H] = sum_a w[b,a] * preds[e,b,a,h]
            baseline = torch.einsum("ebah,ba->ebh", preds_all, w)

            effect_per_h = f_obs - baseline  # [E, B, H]

        elif mode == "signed_oracle_matched":
            # Khớp công thức oracle: mean_{a in candidates} f(s,a) - f(s,a_obs)
            cand = torch.tensor(
                [a for a in self.candidate_actions if 0 <= a < A],
                dtype=torch.long,
                device=self.device,
            )  # [n_cand]

            if cand.numel() == 0:
                cand = torch.arange(A, dtype=torch.long, device=self.device)

            cand_preds = preds_all[:, :, cand, :]      # [E, B, n_cand, H]
            cand_mean = cand_preds.mean(dim=2)         # [E, B, H]

            effect_per_h = cand_mean - f_obs           # [E, B, H]

        elif mode == "range":
            # Pieroth impact sample: max_a f - min_a f  (luôn >= 0)
            max_f = preds_all.max(dim=2).values        # [E, B, H]
            min_f = preds_all.min(dim=2).values        # [E, B, H]
            effect_per_h = max_f - min_f               # [E, B, H]

        elif mode == "mean_abs":
            # Bản v1: mean_{a != a_obs} |f(a) - f(a_obs)|
            diff = torch.abs(preds_all - f_obs.unsqueeze(2))  # [E, B, A, H]

            mask = torch.ones(B, A, dtype=torch.float32, device=self.device)
            mask.scatter_(1, idx_obs.view(B, 1), 0.0)          # [B, A], 0 tại a_obs

            denom = torch.clamp(mask.sum(dim=1), min=1.0)      # [B]

            effect_per_h = (
                torch.einsum("ebah,ba->ebh", diff, mask) / denom.view(1, B, 1)
            )  # [E, B, H]

        else:
            raise ValueError(f"mode không hợp lệ: {mode}")

        # ---------------------------------------------------------------
        # DOUBLY ROBUST CORRECTION
        # Chỉ áp cho các mode CÓ DẤU. Với "range" thì DR không có nghĩa
        # (range không phải một kỳ vọng tuyến tính nên không có dạng DR đóng).
        # ---------------------------------------------------------------
        apply_dr = (
            self.use_doubly_robust
            and mode in ("signed_aristocrat", "signed_oracle_matched")
            and observed_returns is not None
            and behaviour_probs_obs is not None
        )

        if apply_dr:
            R_obs = torch.tensor(
                np.asarray(observed_returns, dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )  # [B]

            b_obs = torch.tensor(
                np.asarray(behaviour_probs_obs, dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            ).clamp(min=1.0 / self.iw_clip, max=1.0)  # [B]

            # residual đo ở horizon cuối (R_obs là R^(H))
            residual = R_obs.view(1, B) - f_obs[:, :, -1]  # [E, B]

            # trọng số hiệu chỉnh: (1/b - 1), clip để không nổ phương sai
            iw_minus_one = torch.clamp(
                1.0 / b_obs - 1.0, min=0.0, max=self.iw_clip
            )  # [B]

            correction = residual * iw_minus_one.view(1, B)  # [E, B]

            if mode == "signed_oracle_matched":
                # ở mode này effect = baseline_cand - f_obs, tức f_obs mang dấu âm
                correction = -correction

            # Chỉ cộng vào horizon cuối (horizon khác không có observed target
            # tương ứng ở đây -> giữ nguyên plug-in cho chúng).
            effect_per_h = effect_per_h.clone()
            effect_per_h[:, :, -1] = effect_per_h[:, :, -1] + correction

            dr_correction = torch.mean(torch.abs(correction), dim=0)  # [B]

        return {
            "effect": effect_per_h[:, :, -1],   # [E, B]
            "effect_per_h": effect_per_h,       # [E, B, H]
            "dr_correction": dr_correction,     # [B]
        }

    # =====================================================================
    # API scoring
    # =====================================================================

    def score_batch(
        self,
        obs_i_batch,
        action_i_batch,
        observed_action_j_batch,
        z_core_excl_j_batch,
        m_periph_excl_j_batch,
        belief_summary_batch,
        # ---- optional, mới ở v2 ----
        policy_probs_j_batch=None,
        observed_returns_batch=None,
        behaviour_probs_obs_batch=None,
    ):
        """
        GIỮ NGUYÊN chữ ký v1 -> runner cũ gọi được ngay.

        Returns:
            mu_arr:    np.ndarray [B]  — CÓ DẤU nếu effect_mode là signed_*
            sigma_arr: np.ndarray [B]  — std across ensemble (bất định epistemic)
        """
        out = self.score_batch_full(
            obs_i_batch=obs_i_batch,
            action_i_batch=action_i_batch,
            observed_action_j_batch=observed_action_j_batch,
            z_core_excl_j_batch=z_core_excl_j_batch,
            m_periph_excl_j_batch=m_periph_excl_j_batch,
            belief_summary_batch=belief_summary_batch,
            policy_probs_j_batch=policy_probs_j_batch,
            observed_returns_batch=observed_returns_batch,
            behaviour_probs_obs_batch=behaviour_probs_obs_batch,
        )

        return out["mu"], out["sigma"]

    def score_batch_full(
        self,
        obs_i_batch,
        action_i_batch,
        observed_action_j_batch,
        z_core_excl_j_batch,
        m_periph_excl_j_batch,
        belief_summary_batch,
        policy_probs_j_batch=None,
        observed_returns_batch=None,
        behaviour_probs_obs_batch=None,
    ) -> Dict[str, np.ndarray]:
        """
        Phiên bản đầy đủ — cung cấp mọi thứ influence_signature.py cần.

        Returns dict of np.ndarray:
            mu            [B]    effect trung bình qua ensemble (CÓ DẤU)
            sigma         [B]    std qua ensemble = bất định epistemic
            mu_per_h      [B, H] effect theo từng horizon -> chiều LATENCY
            latency       [B]    trọng tâm horizon của |effect|, trong [0, H-1]
            mu_range      [B]    impact kiểu Pieroth (luôn >= 0) -> baseline
            dr_correction [B]    độ lớn hiệu chỉnh DR (chẩn đoán bias model)
        """
        B = len(obs_i_batch)

        if B == 0:
            z = np.zeros((0,), dtype=np.float32)
            return {
                "mu": z,
                "sigma": z,
                "mu_per_h": np.zeros((0, self.n_horizons), dtype=np.float32),
                "latency": z,
                "mu_range": z,
                "dr_correction": z,
            }

        obs = np.asarray(obs_i_batch, dtype=np.float32)
        z_arr = np.asarray(z_core_excl_j_batch, dtype=np.float32)
        m_arr = np.asarray(m_periph_excl_j_batch, dtype=np.float32)
        belief = np.asarray(belief_summary_batch, dtype=np.float32)
        a_i = np.asarray(action_i_batch, dtype=np.int64).reshape(-1)
        a_j = np.asarray(observed_action_j_batch, dtype=np.int64).reshape(-1)

        # MỘT forward pass duy nhất cho mọi hành động thay thế.
        preds_all = self._predict_all_actions(
            obs=obs, action_i=a_i, z=z_arr, m=m_arr, belief=belief
        )  # [E, B, A, H]

        # ---- effect theo mode chính ------------------------------------
        res = self._compute_effects(
            preds_all=preds_all,
            action_j_obs=a_j,
            policy_probs_j=policy_probs_j_batch,
            observed_returns=observed_returns_batch,
            behaviour_probs_obs=behaviour_probs_obs_batch,
            mode=self.effect_mode,
        )

        effect = res["effect"]              # [E, B]
        effect_per_h = res["effect_per_h"]  # [E, B, H]

        mu = torch.mean(effect, dim=0)      # [B]

        if effect.shape[0] <= 1:
            sigma = torch.zeros_like(mu)
        else:
            sigma = torch.sqrt(
                torch.var(effect, dim=0, unbiased=True) + self.eps
            )  # [B]

        mu_per_h = torch.mean(effect_per_h, dim=0)  # [B, H]

        # ---- LATENCY: trọng tâm horizon của |effect| --------------------
        # latency = sum_h h * |e_h| / sum_h |e_h|
        # Blocker (tức thì) -> gần 0. Relay (trễ) -> gần H-1.
        abs_h = torch.abs(mu_per_h)                              # [B, H]
        h_idx = torch.arange(
            self.n_horizons, dtype=torch.float32, device=self.device
        ).view(1, -1)                                            # [1, H]

        denom = torch.clamp(abs_h.sum(dim=1), min=self.eps)      # [B]
        latency = (abs_h * h_idx).sum(dim=1) / denom             # [B]

        # ---- mu_range: luôn tính, dùng làm baseline Pieroth -------------
        res_range = self._compute_effects(
            preds_all=preds_all,
            action_j_obs=a_j,
            mode="range",
        )
        mu_range = torch.mean(res_range["effect"], dim=0)        # [B]

        self.latest_dr_correction_magnitude = float(
            torch.mean(res["dr_correction"]).cpu().item()
        )

        to_np = lambda t: t.detach().cpu().numpy().astype(np.float32)

        return {
            "mu": to_np(mu),
            "sigma": to_np(sigma),
            "mu_per_h": to_np(mu_per_h),
            "latency": to_np(latency),
            "mu_range": to_np(mu_range),
            "dr_correction": to_np(res["dr_correction"]),
        }

    def score_pair(
        self,
        obs_i,
        action_i,
        observed_action_j,
        z_core_excl_j,
        m_periph_excl_j,
        belief_summary,
        **kwargs,
    ):
        """Wrapper một cặp — giữ nguyên chữ ký v1."""
        mu_arr, sigma_arr = self.score_batch(
            obs_i_batch=[obs_i],
            action_i_batch=[int(action_i)],
            observed_action_j_batch=[int(observed_action_j)],
            z_core_excl_j_batch=[z_core_excl_j],
            m_periph_excl_j_batch=[m_periph_excl_j],
            belief_summary_batch=[belief_summary],
            **kwargs,
        )

        if len(mu_arr) == 0:
            return 0.0, 0.0

        return float(mu_arr[0]), float(sigma_arr[0])

    # =====================================================================
    # Chẩn đoán
    # =====================================================================

    def get_diagnostics(self) -> Dict[str, float]:
        """
        Số liệu chẩn đoán. Ba con số quan trọng nhất:

        ensemble_disagreement:
            Nếu ~ 0 -> ensemble đang GIẢ, sigma vô nghĩa (bệnh của v1).
            Kỳ vọng sau khi vá: > 0 và giảm dần khi học tốt lên.

        dr_correction_magnitude:
            Đo mức độ reward model bị lệch. Lớn -> model sai nhiều,
            DR đang gánh. Nhỏ -> model tốt.
            Vẽ theo thời gian là một hình đẹp cho paper.

        interventional_fraction:
            Tỷ lệ mẫu đến từ eps-forcing (can thiệp thật). Reviewer sẽ hỏi.
        """
        n = max(1, len(self.buffer))

        return {
            "buffer_size": int(len(self.buffer)),
            "n_interventional_samples": int(self.n_interventional_samples),
            "interventional_fraction": float(self.n_interventional_samples) / float(n),
            "latest_loss": float(self.latest_loss),
            "latest_train_residual": float(self.latest_train_residual),
            "latest_holdout_residual": float(self.latest_holdout_residual),
            "ensemble_disagreement": float(self.latest_ensemble_disagreement),
            "dr_correction_magnitude": float(self.latest_dr_correction_magnitude),
            "effect_mode": self.effect_mode,
            "use_doubly_robust": bool(self.use_doubly_robust),
            "n_ensemble": int(self.n_ensemble),
            "n_horizons": int(self.n_horizons),
        }


# Alias để runner cũ import không gãy.
LocalCounterfactualProxyEnsemble = LocalCounterfactualProxyEnsemble
