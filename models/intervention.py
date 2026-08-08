"""
intervention.py — ε-forced action controller cho CIG-AMF v2.

=============================================================================
VÌ SAO CÓ FILE NÀY
=============================================================================
Bản v1 định nghĩa w_ij bằng do-operator (Eq. 3-4 trong paper) nhưng ước lượng
nó bằng một reward model thuần quan sát (Eq. 7). Vấn đề: a_j KHÔNG hề ngẫu
nhiên — nó tương quan với s, a_i, và hành động của mọi neighbour khác. Nên
hiệu hai lần dự đoán của model có thể chỉ là correlation đội lốt causation,
và paper buộc phải viết câu tự thú "không claim w_hat = w".

File này sửa tận gốc: với xác suất epsilon, ta ÉP agent j chọn hành động
ngẫu nhiên đều thay vì theo policy. Khi a_j được randomize thật, mọi đường
confounding bị CẮT ĐỨT VỀ MẶT CƠ HỌC — đó đúng nghĩa do(a_j), không phải
xấp xỉ. Đây là randomized controlled trial mà y học phải tốn triệu đô, còn
ta làm miễn phí vì ta sở hữu simulator.

=============================================================================
HỆ QUẢ QUAN TRỌNG: PROPENSITY TRỞ NÊN CHÍNH XÁC
=============================================================================
Behaviour policy hiệu dụng của agent j tại bước t là một hỗn hợp:

    b_j(a | s) = eps * (1/|A|)  +  (1 - eps) * pi_j(a | s)      nếu j đang bị ép
    b_j(a | s) = pi_j(a | s)                                     nếu j không bị ép

Ta BIẾT CHÍNH XÁC b_j (vì ta tự train pi_j và tự chọn eps). Trong off-policy
evaluation, propensity thường phải ước lượng và đó là nguồn sai số chính.
Ở đây nó exact → doubly-robust estimator (structural_proxy_v2.py) chỉ cần
"một trong hai mô hình đúng" và ta ĐÃ CÓ SẴN một cái đúng tuyệt đối.

Ngoài ra eps-forcing đảm bảo luôn giả định POSITIVITY / OVERLAP:
    b_j(a | s) >= eps / |A| > 0  cho mọi a
Không có nó, hành động phản thực a'_j mà j không bao giờ làm sẽ khiến model
phải ngoại suy vào vùng không có dữ liệu → ước lượng vô nghĩa. Bản v1 KHÔNG
nêu giả định này ở đâu cả.
=============================================================================
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class EpsilonForcedActionController:
    """
    Quản lý việc ép hành động ngẫu nhiên rải rác trong lúc training.

    Cách dùng trong runner (thay cho chỗ đang sample action từ policy):

        forced_mask, effective_probs = controller.apply(
            actions=actions,             # list[int] length n_agents, từ policy
            policy_probs=policy_probs,   # np [n_agents, action_dim], từ softmax logits
        )
        # actions đã được sửa in-place tại các vị trí bị ép
        obs, rew, done, info = env.step(actions)

    Rồi khi push replay:
        step["forced_mask"]     = forced_mask       # np bool [n_agents]
        step["behaviour_probs"] = effective_probs   # np [n_agents, action_dim]

    Args:
        n_agents:
            số agent trong population.
        action_dim:
            kích thước không gian hành động rời rạc.
        eps:
            xác suất mỗi agent bị ép ở mỗi bước. 0.02-0.05 là hợp lý:
            đủ để tích luỹ mẫu can thiệp, đủ nhỏ để không phá reward.
        max_forced_per_step:
            trần số agent bị ép cùng lúc. Ép nhiều agent một lúc làm
            nhiễu lẫn nhau và khó quy trách nhiệm, nên giới hạn.
            None = không giới hạn.
        anneal_to:
            nếu khác None, eps sẽ giảm tuyến tính từ eps xuống anneal_to
            theo anneal_episodes. Dùng khi muốn can thiệp mạnh lúc đầu
            (proxy cần dữ liệu) rồi nhẹ dần (bảo vệ reward cuối).
        anneal_episodes:
            số episode để anneal.
        rng:
            np.random.RandomState để tái lập được theo seed.
    """

    def __init__(
        self,
        n_agents: int,
        action_dim: int,
        eps: float = 0.03,
        max_forced_per_step: Optional[int] = None,   # [FIX-P1] was 2
        anneal_to: Optional[float] = None,
        anneal_episodes: int = 60,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)

        self.eps_initial = float(eps)
        self.eps = float(eps)

        # [FIX-P1] None = không cap (mặc định mới). Ép về int nếu có giá trị
        # để `cfg.get(..., 2)` trả về None từ config không bị hiểu nhầm.
        self.max_forced_per_step = (
            None if max_forced_per_step is None else int(max_forced_per_step)
        )
        self.anneal_to = anneal_to
        self.anneal_episodes = int(max(1, anneal_episodes))

        self.rng = rng if rng is not None else np.random.RandomState(0)

        self.episode = 0

        # eps riêng cho từng agent (None = dùng eps chung cho tất cả)
        self._eps_per_agent = None

        # Thống kê để báo cáo trong paper (reviewer sẽ hỏi "eps làm mất bao nhiêu reward")
        self.total_steps = 0
        self.total_forced = 0

    # ------------------------------------------------------------------
    # Lịch trình eps
    # ------------------------------------------------------------------

    def step_episode(self):
        """Gọi một lần sau mỗi episode để cập nhật lịch trình anneal."""
        self.episode += 1

        if self.anneal_to is None:
            return

        frac = min(1.0, float(self.episode) / float(self.anneal_episodes))
        self.eps = float(
            (1.0 - frac) * self.eps_initial + frac * float(self.anneal_to)
        )

    def get_eps(self) -> float:
        return float(self.eps)

    # ------------------------------------------------------------------
    # Áp dụng can thiệp
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # NHẮM MỤC TIÊU CAN THIỆP THEO BẤT ĐỊNH
    # ------------------------------------------------------------------

    def set_priority(
        self,
        scores: Optional[np.ndarray],
        floor_ratio: float = 0.34,
    ):
        """
        Phân bổ ngân sách can thiệp theo mức "chưa hiểu" của từng agent.

        VÌ SAO CẦN
        ----------
        Ta đang trả 3% reward để mua dữ liệu nhân quả. Nếu rải đều, phần lớn
        3% đó rơi vào những cặp đã hiểu rõ từ lâu — tiền tiêu mà không mua
        thêm được thông tin gì.

        Ví dụ: bạn có 3 phiếu hỏi ý kiến chuyên gia. Đem hỏi 3 câu bạn đã
        biết đáp án thì phí. Hỏi đúng 3 câu đang mơ hồ nhất mới đáng.

        Ở đây "đang mơ hồ nhất" = epistemic uncertainty cao = ensemble đang
        bất đồng về agent đó.

        RÀNG BUỘC KHÔNG ĐƯỢC VI PHẠM
        ----------------------------
        Mọi agent PHẢI giữ xác suất bị ép > 0. Nếu một agent không bao giờ
        được ép, giả định overlap (A2) sụp với agent đó, và mọi ước lượng
        nhân quả liên quan tới nó mất căn cứ. Vì thế có `floor_ratio`: mỗi
        agent luôn giữ ít nhất `floor_ratio` phần ngân sách trung bình.

        TÍNH HỢP LỆ NHÂN QUẢ VẪN GIỮ NGUYÊN
        -----------------------------------
        Việc CHỌN AI để ép có thể phụ thuộc vào bất cứ thứ gì (kể cả lịch sử,
        bất định...) mà không phá tính nhân quả, MIỄN LÀ:
          (a) hành động sau khi đã quyết định ép vẫn bốc ngẫu nhiên đều,
              độc lập với trạng thái  -> giữ nguyên do(a_j)
          (b) xác suất ép của từng agent được GHI LẠI và đưa vào propensity
              -> DR vẫn đúng
        Cả hai đều được bảo đảm bên dưới: eps_per_agent được ghi vào
        effective_probs trả về.

        Args:
            scores: np [n_agents], càng cao càng ưu tiên ép. None = tắt.
            floor_ratio: sàn ngân sách, trong (0, 1].
        """
        if scores is None:
            self._eps_per_agent = None
            return

        s = np.asarray(scores, dtype=np.float64).reshape(-1)

        if s.shape[0] != self.n_agents:
            raise ValueError(
                f"scores phải có length {self.n_agents}, nhận {s.shape[0]}"
            )

        s = np.clip(s, 0.0, None)
        total = float(s.sum())

        if total <= 1e-12:
            self._eps_per_agent = None
            return

        # Chuẩn hoá thành trọng số trung bình 1.0, rồi trộn với sàn.
        w = s / (total / self.n_agents)
        fr = float(np.clip(floor_ratio, 1e-3, 1.0))
        w = fr + (1.0 - fr) * w

        # Giữ NGÂN SÁCH TỔNG không đổi: mean(eps_j) = eps.
        w = w / float(np.mean(w))

        eps_j = np.clip(self.eps * w, 0.0, 1.0)

        self._eps_per_agent = eps_j.astype(np.float64)

    def get_eps_per_agent(self) -> np.ndarray:
        """np [n_agents] — xác suất ép hiện tại của từng agent."""
        if getattr(self, "_eps_per_agent", None) is None:
            return np.full(self.n_agents, self.eps, dtype=np.float64)

        return self._eps_per_agent.copy()

    def apply(
        self,
        actions: List[int],
        policy_probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ép ngẫu nhiên một số agent, và trả về propensity hiệu dụng.

        Args:
            actions:
                list[int] length n_agents. Hành động đã sample từ policy.
                SẼ BỊ SỬA IN-PLACE tại các vị trí bị ép.
            policy_probs:
                np.ndarray shape [n_agents, action_dim].
                pi_j(a | s) cho mọi agent, mọi hành động.

        Returns:
            forced_mask:
                np.ndarray bool shape [n_agents]. True = agent này bị ép.
            effective_probs:
                np.ndarray float32 shape [n_agents, action_dim].
                b_j(a | s) — behaviour policy hiệu dụng, dùng cho DR estimator.
        """
        probs = np.asarray(policy_probs, dtype=np.float32)

        if probs.shape != (self.n_agents, self.action_dim):
            raise ValueError(
                f"policy_probs phải có shape [{self.n_agents}, {self.action_dim}], "
                f"nhận được {probs.shape}"
            )

        # ---- 1. Quyết định agent nào bị ép -------------------------------
        # eps_vec: [n_agents] — bằng nhau nếu không set_priority,
        # khác nhau nếu đang nhắm mục tiêu theo bất định.
        eps_vec = self.get_eps_per_agent()          # [n_agents]

        draw = self.rng.rand(self.n_agents)         # [n_agents]
        forced_mask = draw < eps_vec                # [n_agents] bool

        # ------------------------------------------------------------------
        # [FIX-P1] Giới hạn số agent bị ép cùng lúc — PHÁ VỠ "propensity known
        # exactly", claim trung tâm của paper (§B3: b_j "is known exactly,
        # since pi_j is the learner's own network and eps is a chosen
        # constant").
        #
        # Cap là một phép LỌC PHỤ THUỘC KẾT QUẢ BỐC THĂM của toàn quần thể:
        # xác suất j thực sự bị ép không còn là eps_j nữa mà là
        #     eps_eff_j = eps_j * P(j sống sót qua cap)
        # trong đó P(...) phụ thuộc số agent khác cũng trúng ở bước đó.
        # Với eps=0.03, n=24, cap=2: X ~ Bin(24, 0.03), P(X>2) ~ 3.8%, nên
        # eps_eff ~ 0.0282 — propensity đang bị KHAI CAO ~6% một cách hệ
        # thống, và DR chia sai đúng bằng tỉ lệ đó (mất tính không chệch).
        #
        # Cap gần như vô dụng ở chế độ mặc định: E[X] = 24*0.03 = 0.72 agent
        # mỗi bước, tức cap=2 hầu như không bao giờ chạm. Nó trả giá bằng tính
        # đúng đắn của claim để đổi lấy gần như không có gì.
        #
        # => Mặc định TẮT cap (max_forced_per_step=None). Nếu ai đó bật lại,
        # cảnh báo một lần cho biết propensity ghi lại chỉ còn là xấp xỉ.
        # ------------------------------------------------------------------
        if self.max_forced_per_step is not None:
            forced_ids = np.flatnonzero(forced_mask)

            if len(forced_ids) > int(self.max_forced_per_step):
                if not getattr(self, "_cap_warned", False):
                    print(
                        "[eps-forcing][WARN] max_forced_per_step="
                        f"{self.max_forced_per_step} vừa CẮT bớt số agent bị ép. "
                        "Propensity ghi lại (eps danh nghĩa) giờ LỚN HƠN xác "
                        "suất ép thực tế => DR chệch có hệ thống, và claim "
                        "'b_j known exactly' của paper không còn đúng. "
                        "Đặt forcer_max_forced_per_step=None để khôi phục."
                    )
                    self._cap_warned = True

                keep = self.rng.choice(
                    forced_ids,
                    size=int(self.max_forced_per_step),
                    replace=False,
                )
                forced_mask = np.zeros(self.n_agents, dtype=bool)
                forced_mask[keep] = True

        # ---- 2. Ép hành động ngẫu nhiên đều ------------------------------
        for j in np.flatnonzero(forced_mask):
            actions[int(j)] = int(self.rng.randint(0, self.action_dim))

        # ---- 3. Tính propensity hiệu dụng --------------------------------
        # Với agent KHÔNG bị ép:  b = pi
        # Với agent BỊ ép:        b = eps * uniform + (1-eps) * pi
        #
        # Lưu ý tinh tế: ta dùng công thức mixture cho agent bị ép chứ không
        # phải uniform thuần. Lý do: từ góc nhìn của estimator, quá trình sinh
        # dữ liệu là "với xác suất eps thì uniform, ngược lại thì pi" — đó là
        # marginal distribution của hành động, và đó mới là propensity đúng
        # để chia trong importance weighting.
        uniform = np.full(
            (self.n_agents, self.action_dim),
            1.0 / float(self.action_dim),
            dtype=np.float32,
        )  # [n_agents, action_dim]

        # Mỗi agent có eps riêng -> propensity riêng. Ghi lại đúng eps đã
        # dùng, nếu không DR sẽ chia sai và mất tính không chệch.
        e = eps_vec.reshape(-1, 1).astype(np.float32)   # [n_agents, 1]

        effective_probs = (
            e * uniform + (1.0 - e) * probs
        ).astype(np.float32)  # [n_agents, action_dim]

        # Chuẩn hoá lại phòng sai số dấu phẩy động.
        row_sum = np.sum(effective_probs, axis=1, keepdims=True)  # [n_agents, 1]
        effective_probs = effective_probs / np.clip(row_sum, 1e-8, None)

        # ---- 4. Thống kê --------------------------------------------------
        self.total_steps += self.n_agents
        self.total_forced += int(np.sum(forced_mask))

        return forced_mask, effective_probs

    # ------------------------------------------------------------------
    # Báo cáo
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, float]:
        """Số liệu để đưa vào bảng trong paper."""
        rate = (
            float(self.total_forced) / float(self.total_steps)
            if self.total_steps > 0
            else 0.0
        )

        return {
            "eps_current": float(self.eps),
            "eps_initial": float(self.eps_initial),
            "total_agent_steps": int(self.total_steps),
            "total_forced": int(self.total_forced),
            "realised_forcing_rate": float(rate),
            "episode": int(self.episode),
            "targeting_enabled": bool(
                getattr(self, "_eps_per_agent", None) is not None
            ),
            "eps_min_agent": float(np.min(self.get_eps_per_agent())),
            "eps_max_agent": float(np.max(self.get_eps_per_agent())),
        }


class OracleInterventionSampler:
    """
    Wrapper quanh clone/restore của env để lấy w_ij NHÂN QUẢ THẬT.

    Env đã có sẵn:
        clone_state(), restore_state(state),
        rollout_from_current_state(forced={j: action}, horizon, discount)

    Dùng cho hai việc:
      (1) Exp3 tiny-oracle calibration — so proxy với ground truth.
      (2) Sinh một lượng nhỏ mẫu can thiệp CHUẨN XÁC để anchor proxy
          (khác eps-forcing ở chỗ: cái này giữ nguyên trạng thái, đổi đúng
          1 hành động, so sánh trực tiếp — tức tầng 3 counterfactual thật).

    CẢNH BÁO ĐỘ KHỚP ESTIMAND (lỗi trong bản v1):
        Bản v1 để proxy tính  mean_a |f(a) - f(a_obs)|   (LUÔN >= 0)
        còn oracle tính        mean_a  (R(a) - R_base)    (CÓ DẤU)
        Hai đại lượng này KHÔNG THỂ khớp nhau. Một neighbour có ảnh hưởng
        đối xứng (giúp ở action này, hại ở action kia) cho oracle ~ 0 nhưng
        proxy ~ lớn. Đó là lý do Exp3 không bao giờ ra kết quả tốt.
        Class này expose CẢ HAI dạng để so đúng cặp với nhau.
    """

    def __init__(
        self,
        env,
        horizon: int = 3,
        discount: float = 0.95,
        n_trials: int = 3,
    ):
        self.env = env
        self.horizon = int(horizon)
        self.discount = float(discount)
        self.n_trials = int(max(1, n_trials))

    def signed_effect(
        self,
        ego_id: int,
        neighbor_id: int,
        candidate_actions: Optional[List[int]] = None,
    ) -> float:
        """
        w_ij CÓ DẤU (khớp với 'signed_oracle_matched' của proxy v2).

            w = mean_a [ R_i(do(a_j = a)) ] - R_i(baseline)

        w > 0: ép j làm khác đi thì i TỐT hơn  -> j hiện đang cản i
        w < 0: ép j làm khác đi thì i TỆ hơn   -> j hiện đang giúp i

        Returns:
            float
        """
        return float(
            self.env.compute_oracle_influence_from_current_state(
                ego_id=int(ego_id),
                agent_j=int(neighbor_id),
                intervention_action=(
                    None
                    if candidate_actions is None
                    else int(candidate_actions[0])
                ),
                horizon=self.horizon,
                n_trials=self.n_trials,
                discount=self.discount,
            )
        )

    def range_effect(
        self,
        ego_id: int,
        neighbor_id: int,
        candidate_actions: List[int],
    ) -> float:
        """
        Impact kiểu Pieroth (ICML 2024), Definition 5.1:

            U^{j->i} = max_a R_i(do(a_j=a)) - min_a R_i(do(a_j=a))

        Luôn >= 0. Đây là BASELINE ĐỐI CHỨNG quan trọng: nếu signature CÓ DẤU
        của ta phân loại vai trò tốt hơn đại lượng range KHÔNG DẤU này, đó là
        bằng chứng trực tiếp cho novelty (Pieroth chủ động tránh counterfactual
        và không có dấu).

        Returns:
            float >= 0
        """
        saved = self.env.clone_state()
        returns_per_action = []

        try:
            for action in candidate_actions:
                trial_vals = []

                for _ in range(self.n_trials):
                    self.env.restore_state(saved)
                    out = self.env.rollout_from_current_state(
                        forced={int(neighbor_id): int(action)},
                        horizon=self.horizon,
                        discount=self.discount,
                    )
                    trial_vals.append(float(out[int(ego_id)]))

                returns_per_action.append(float(np.mean(trial_vals)))
        finally:
            self.env.restore_state(saved)

        if len(returns_per_action) == 0:
            return 0.0

        return float(max(returns_per_action) - min(returns_per_action))

    def full_profile(
        self,
        ego_id: int,
        neighbor_id: int,
        candidate_actions: List[int],
    ) -> Dict[str, float]:
        """
        Trả về ground-truth profile đầy đủ cho một cặp — dùng làm nhãn
        chuẩn khi đánh giá influence signature.

        Returns:
            dict với các khoá:
                signed   : mean_a R(a) - R_base            (có dấu)
                range    : max_a R(a) - min_a R(a)         (Pieroth-style, >=0)
                best     : max_a R(a) - R_base             (j có thể giúp i tối đa bao nhiêu)
                worst    : min_a R(a) - R_base             (j có thể hại i tối đa bao nhiêu)
                spread   : std_a R(a)                      (độ phân tán theo hành động)
        """
        saved = self.env.clone_state()

        try:
            base_vals = []
            for _ in range(self.n_trials):
                self.env.restore_state(saved)
                out = self.env.rollout_from_current_state(
                    forced=None,
                    horizon=self.horizon,
                    discount=self.discount,
                )
                base_vals.append(float(out[int(ego_id)]))

            base = float(np.mean(base_vals))

            per_action = []
            for action in candidate_actions:
                trial_vals = []
                for _ in range(self.n_trials):
                    self.env.restore_state(saved)
                    out = self.env.rollout_from_current_state(
                        forced={int(neighbor_id): int(action)},
                        horizon=self.horizon,
                        discount=self.discount,
                    )
                    trial_vals.append(float(out[int(ego_id)]))

                per_action.append(float(np.mean(trial_vals)))
        finally:
            self.env.restore_state(saved)

        if len(per_action) == 0:
            return {
                "signed": 0.0,
                "range": 0.0,
                "best": 0.0,
                "worst": 0.0,
                "spread": 0.0,
            }

        arr = np.asarray(per_action, dtype=np.float64)  # [n_candidates]

        return {
            "signed": float(np.mean(arr) - base),
            "range": float(np.max(arr) - np.min(arr)),
            "best": float(np.max(arr) - base),
            "worst": float(np.min(arr) - base),
            "spread": float(np.std(arr)),
        }
