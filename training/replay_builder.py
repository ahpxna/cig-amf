import numpy as np

from models.structural_proxy import build_pair_feat  # [FIX-X1]


class MultiEgoReplayBuilder:
    """
    Xây dữ liệu supervised cho proxy ensemble từ trajectory toàn population.

    Bám đúng conditioning set của paper:

        r_hat_ij(
            s_i,
            a_i,
            a_j,
            Z_i^{-j},
            M_i^{-j},
            B_i
        ) -> R_i^(H)

    Mỗi timestep t, ego i, neighbor j:
        sample_ij^t = (
            obs_i^t,
            a_i^t,
            a_j^t,
            z_{i,C_i \\ {j}}^t,
            m_{i,P_i \\ {j}}^t,
            b_i^t,
            R_i^{(H)}(t)
        )

    Quan trọng:
    - File này không tự recompute context.
    - Context excluding-j lấy từ trajectory cache tại thời điểm action selection.
    - Như vậy proxy sample khớp với state/context thật lúc policy ra action.
    - Hàm push_trajectory_to_proxy() phải được gọi every episode, kể cả Stage 0.
    - Stage 0 không train/update belief, nhưng phải collect proxy buffer.
    """

    def __init__(self, discount=0.95, horizon=8):
        # P2 (Phần 2.2): H_causal >= max_latency + 2. Với 4 latency tier của
        # Omni-Arena (blocker h=1, gatekeeper h=2-3, relay h=4-5,
        # controller h=6+), H >= 8. Default bump 3 -> 8 (caller vẫn có thể
        # truyền horizon= khác nếu cần).
        self.discount = float(discount)
        self.horizon = int(horizon)

    def build_h_step_returns(self, trajectory, n_agents):
        """
        R_i^(H)(t) = sum_{h=0}^{H-1} gamma^h r_i(t+h)

        Args:
            trajectory:
                list step dict.
                Mỗi step phải có key "rewards".
            n_agents:
                số agents trong population.

        Return:
            list length T.
            out[t][ego] = H-step return của ego tại timestep t.
        """
        T = len(trajectory)
        out = []

        for t in range(T):
            row = {}

            for ego in range(int(n_agents)):
                val = 0.0

                for h in range(self.horizon):
                    if t + h < T:
                        val += (
                            self.discount ** h
                        ) * float(trajectory[t + h]["rewards"][ego])

                row[ego] = float(val)

            out.append(row)

        return out

    def build_h_step_returns_multi(self, trajectory, n_agents, n_horizons=None):
        """
        P2: bản multi-horizon của build_h_step_returns() -- trả về
        R_i^(1), R_i^(2), ..., R_i^(n_horizons) cho mỗi (t, ego), tức
        target_returns_multi thật (không phải broadcast một số) để nạp cho
        structural_proxy_v2 (n_horizons=8 theo Phần 2.2).

        R_i^(h)(t) = sum_{k=0}^{h-1} gamma^k * r_i(t+k)  (h = 1..n_horizons)

        Return:
            list length T. out[t][ego] = np.ndarray shape [n_horizons].
        """
        if n_horizons is None:
            n_horizons = self.horizon

        T = len(trajectory)
        out = []

        for t in range(T):
            row = {}
            for ego in range(int(n_agents)):
                cum = 0.0
                per_h = np.zeros((int(n_horizons),), dtype=np.float32)
                for h in range(int(n_horizons)):
                    if t + h < T:
                        cum += (self.discount ** h) * float(trajectory[t + h]["rewards"][ego])
                    per_h[h] = cum
                row[ego] = per_h
            out.append(row)

        return out

    def _require_step_fields(self, step):
        required = [
            "obs_all",
            "actions",
            "rewards",
            "core_context_excluding",
            "periph_context_excluding",
            "belief_summary_cache",
        ]

        missing = [k for k in required if k not in step]

        if len(missing) > 0:
            raise KeyError(
                "MultiEgoReplayBuilder expected trajectory step to contain "
                f"{required}, but missing {missing}. "
                "Runner must cache context at action-selection time."
            )

    def push_trajectory_to_proxy(
        self,
        trajectory,
        proxy_ensemble,
        env,
    ):
        """
        Đẩy toàn bộ population-wide trajectory vào proxy buffer.

        Required fields trong mỗi step:
            obs_all
            actions
            rewards
            core_context_excluding
            periph_context_excluding
            belief_summary_cache

        Return:
            số supervised samples đã push.

        Lưu ý rất quan trọng:
        - Hàm này phải được gọi mọi episode, kể cả Stage 0.
        - Stage 0 không train/update belief, nhưng phải collect proxy buffer.
        - Hàm này không được dùng scheduler.should_update_graph() để quyết định collect.
        """
        if trajectory is None or len(trajectory) == 0:
            return 0

        n_agents = int(env.n_agents)
        h_returns = self.build_h_step_returns(trajectory, n_agents)
        # P2: target_returns_multi thật (n_horizons phần tử -- mặc định 8),
        # lấy trực tiếp từ trajectory thay vì để add_sample() broadcast một
        # số vô hướng ra mọi horizon (fallback cũ, kém chính xác hơn).
        n_horizons_for_push = getattr(proxy_ensemble, "n_horizons", self.horizon)
        h_returns_multi = self.build_h_step_returns_multi(
            trajectory, n_agents, n_horizons=n_horizons_for_push
        )

        pushed = 0

        for t, step in enumerate(trajectory):
            self._require_step_fields(step)

            obs_all = step["obs_all"]
            actions = step["actions"]

            # [intervention.py — DR] forced_mask/behaviour_probs chỉ có ở
            # final_runner.py (baseline_runner.py không dùng eps-forcing).
            # .get() để KHÔNG crash với trajectory không có hai field này —
            # proxy tự tắt DR về plug-in cho mẫu đó (behaviour_prob_j=None
            # là fallback đã thiết kế sẵn trong add_sample).
            forced_mask = step.get("forced_mask")
            behaviour_probs = step.get("behaviour_probs")

            for ego in range(n_agents):
                obs_i = env.get_obs_of_ego(obs_all, ego)
                action_i = int(actions[ego])

                for j in range(n_agents):
                    if j == ego:
                        continue

                    # [FIX-X1] x_ij tại ĐÚNG timestep t (ảnh chụp hình học
                    # được runner lưu lúc thu thập; env.positions ở đây đã là
                    # state cuối episode nên không dùng được).
                    geom = step.get("geom_snapshot")
                    if geom is None:
                        pair_feat = None
                    else:
                        pair_feat = build_pair_feat(
                            geom["positions"], geom["agent_zone"],
                            geom["grid_size"], geom["n_zones"], ego, j,
                        )

                    z_ex = step["core_context_excluding"][ego][j]
                    m_ex = step["periph_context_excluding"][ego][j]
                    belief_summary = step["belief_summary_cache"][ego]
                    target_h = h_returns[t][ego]
                    target_multi = h_returns_multi[t][ego]

                    was_forced = False
                    behaviour_prob_j = None

                    if forced_mask is not None:
                        was_forced = bool(forced_mask[j])

                    if behaviour_probs is not None:
                        # b_j(a_j_obs | s) — propensity HIỆU DỤNG (đã tính
                        # cả forcing), KHÔNG PHẢI policy_probs thô. Lấy nhầm
                        # sẽ làm DR chệch có hệ thống một cách im lặng.
                        behaviour_prob_j = float(
                            behaviour_probs[j][int(actions[j])]
                        )

                    # [VERIFY-F1b] Đo TẠI ĐIỂM GHI BUFFER: với các mẫu
                    # was_forced=True, a_j phải phân bố ~uniform trên |A|.
                    # Nếu ở runner hist_action_forced đều mà ở đây lệch =>
                    # nhãn bị lệch giữa trajectory và buffer (đúng nghi vấn
                    # "action_j lưu là action TRƯỚC override").
                    if was_forced:
                        _h = getattr(proxy_ensemble, "_vf1b_hist", None)
                        if _h is None:
                            _h = {}
                            proxy_ensemble._vf1b_hist = _h
                        _a = int(actions[j])
                        _h[_a] = _h.get(_a, 0) + 1

                    proxy_ensemble.add_sample(
                        ego_id=ego,
                        neighbor_id=j,
                        obs_i=obs_i,
                        action_i=action_i,
                        observed_action_j=int(actions[j]),
                        z_core_excl_j=z_ex,
                        m_periph_excl_j=m_ex,
                        belief_summary=belief_summary,
                        target_return_h=float(target_h),
                        target_returns_multi=target_multi,
                        behaviour_prob_j=behaviour_prob_j,
                        was_forced=was_forced,
                        pair_feat=pair_feat,   # [FIX-X1]
                    )

                    pushed += 1

        return int(pushed)