import numpy as np


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

    def __init__(self, discount=0.95, horizon=3):
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

        pushed = 0

        for t, step in enumerate(trajectory):
            self._require_step_fields(step)

            obs_all = step["obs_all"]
            actions = step["actions"]

            for ego in range(n_agents):
                obs_i = env.get_obs_of_ego(obs_all, ego)
                action_i = int(actions[ego])

                for j in range(n_agents):
                    if j == ego:
                        continue

                    z_ex = step["core_context_excluding"][ego][j]
                    m_ex = step["periph_context_excluding"][ego][j]
                    belief_summary = step["belief_summary_cache"][ego]
                    target_h = h_returns[t][ego]

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
                    )

                    pushed += 1

        return int(pushed)