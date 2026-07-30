import numpy as np


class BayesLightBeliefState:
    """
    Bayes-light structural belief cho một ego-agent i.

    Với mỗi directed pair (i, j), state giữ:
        b_ij = (mu_bar_ij, sigma_bar_ij, p_core_ij)

    Bám methodology hiện tại của CIG-AMF:
    - Stage 0 dùng weak-prior seeded core để tránh core rỗng.
    - Stage 1 dùng proxy ensemble score để update learned belief.
    - Hysteresis tránh core nhảy ra/vào liên tục.
    - min_core_size tránh learned takeover làm core rỗng hoàn toàn.
    - max_core_size tránh core explosion, ví dụ 24 agents mà 23 neighbours vào core.
    - sigma_floor tránh uncertainty tụt giả về quá thấp khi proxy chưa học tốt.
    - update_batch() phải return (promoted, demoted) để runner warm-start z_ij.

    Công thức update:
        alpha_t = lambda_0 / (1 + uncertainty_scale * sigma_t)

        mu_bar_t = (1 - alpha_t) * mu_bar_{t-1} + alpha_t * mu_t
        sigma_bar_t = (1 - alpha_t) * sigma_bar_{t-1} + alpha_t * sigma_t

        p_core = sigmoid((abs(mu_bar) - tau) / (sigma_bar + eps))

    Hysteresis:
        nếu j chưa ở core: thêm nếu p_core > tau_in
        nếu j đang ở core: giữ nếu p_core >= tau_out

    Capacity control:
        sau hysteresis:
            1. fill tới min_core_size nếu core quá nhỏ.
            2. prune xuống max_core_size nếu core quá lớn.
    """

    def __init__(
        self,
        ego_id,
        neighbor_ids,
        lambda_0=0.12,
        uncertainty_scale=2.0,
        tau=0.10,
        tau_in=0.62,
        tau_out=0.46,
        weak_prior_top_k=2,
        min_core_size=1,
        max_core_size=4,
        sigma_floor=0.05,
        eps=1e-6,
    ):
        self.ego_id = int(ego_id)
        self.neighbor_ids = [int(j) for j in neighbor_ids]

        self.lambda_0 = float(lambda_0)
        self.uncertainty_scale = float(uncertainty_scale)
        self.tau = float(tau)
        self.tau_in = float(tau_in)
        self.tau_out = float(tau_out)

        self.weak_prior_top_k = int(weak_prior_top_k)
        self.min_core_size = int(min_core_size)
        self.max_core_size = int(max_core_size)

        self.sigma_floor = float(sigma_floor)
        self.eps = float(eps)

        if self.min_core_size < 0:
            self.min_core_size = 0

        if self.max_core_size <= 0:
            self.max_core_size = len(self.neighbor_ids)

        self.max_core_size = min(self.max_core_size, len(self.neighbor_ids))

        if self.min_core_size > self.max_core_size:
            self.min_core_size = self.max_core_size

        self.mu_bar = {j: 0.0 for j in self.neighbor_ids}
        self.sigma_bar = {j: 1.0 for j in self.neighbor_ids}
        self.p_core = {j: 0.5 for j in self.neighbor_ids}

        self.core_set = set()
        self.prev_core_set = set()
        self.seeded_core_set = set()

        self.last_promoted = set()
        self.last_demoted = set()
        self.last_core_switch_count = 0

        self.mu_history = {j: [] for j in self.neighbor_ids}
        self.sigma_history = {j: [] for j in self.neighbor_ids}
        self.p_history = {j: [] for j in self.neighbor_ids}
        self.core_history = []

    # ============================================================
    # Basic helpers
    # ============================================================

    def _sigmoid(self, x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    def _safe_sigma(self, sigma):
        return max(float(sigma), self.sigma_floor, self.eps)

    def _priority(self, j):
        """
        Priority dùng cho fill/prune core.

        Dùng abs(mu_bar)/(sigma_bar+eps) để ưu tiên:
        - influence magnitude lớn.
        - uncertainty thấp hơn.

        Có thêm p_core rất nhẹ để tie-break khi mu gần zero.
        """
        j = int(j)

        mu = abs(float(self.mu_bar.get(j, 0.0)))
        sigma = self._safe_sigma(self.sigma_bar.get(j, 1.0))
        p = float(self.p_core.get(j, 0.0))

        return float((mu / (sigma + self.eps)) + 1e-3 * p)

    def _rank_neighbors_by_priority(self, candidates=None):
        if candidates is None:
            candidates = self.neighbor_ids

        candidates = [int(j) for j in candidates if int(j) in self.mu_bar]

        return sorted(
            candidates,
            key=lambda j: self._priority(j),
            reverse=True,
        )

    def _apply_core_capacity_constraints(self, candidate_core):
        """
        Enforce min_core_size và max_core_size.

        Thứ tự:
            1. Fill min để tránh core rỗng.
            2. Prune max để tránh full-core explosion.

        Với n_agents=24, nếu max_core_size=4 thì core tuyệt đối không vượt 4.
        """
        core = set(int(j) for j in candidate_core if int(j) in self.mu_bar)

        if len(core) < self.min_core_size:
            ranked_all = self._rank_neighbors_by_priority(self.neighbor_ids)

            for j in ranked_all:
                core.add(int(j))
                if len(core) >= self.min_core_size:
                    break

        if len(core) > self.max_core_size:
            ranked_core = self._rank_neighbors_by_priority(core)
            core = set(ranked_core[: self.max_core_size])

        return core

    def _record_core_change(self, old_core, new_core):
        old_core = set(old_core)
        new_core = set(new_core)

        self.last_promoted = set(new_core - old_core)
        self.last_demoted = set(old_core - new_core)
        self.last_core_switch_count = len(old_core ^ new_core)

        self.core_history.append(set(new_core))

        if len(self.core_history) > 500:
            self.core_history = self.core_history[-500:]

    # ============================================================
    # Stage 0 seeding
    # ============================================================

    def initialize_from_weak_prior(self, prior_scores):
        """
        Stage 0 seeded core.

        prior_scores:
            dict {neighbor_id: weak structural prior score}

        Lưu ý:
        - Đây không phải ground truth.
        - Đây chỉ là bootstrap để core không rỗng trước khi proxy/belief học đủ.
        - Seeded core cũng bị cap bởi max_core_size.
        """
        ranked = sorted(
            [
                (int(j), float(prior_scores.get(j, 0.0)))
                for j in self.neighbor_ids
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        seed_k = min(
            max(0, int(self.weak_prior_top_k)),
            self.max_core_size,
            len(self.neighbor_ids),
        )

        chosen = [j for j, _ in ranked[:seed_k]]

        old_core = set(self.core_set)

        self.seeded_core_set = set(chosen)
        self.core_set = set(chosen)
        self.prev_core_set = set(chosen)

        for j in self.neighbor_ids:
            if j in self.seeded_core_set:
                self.p_core[j] = 0.58
            else:
                self.p_core[j] = 0.42

        self._record_core_change(old_core, self.core_set)

    def set_fixed_core(self, core_ids):
        """
        Dùng cho ablation hoặc debug nếu cần fixed seeded core.

        Vẫn enforce max_core_size để không tạo full-core ngoài ý muốn.
        """
        old_core = set(self.core_set)

        candidate = set(
            int(j)
            for j in core_ids
            if int(j) in self.neighbor_ids
        )

        candidate = self._apply_core_capacity_constraints(candidate)

        self.core_set = set(candidate)
        self.prev_core_set = set(candidate)
        self.seeded_core_set = set(candidate)

        for j in self.neighbor_ids:
            self.p_core[j] = 0.58 if j in self.core_set else 0.42

        self._record_core_change(old_core, self.core_set)

    # ============================================================
    # Belief update
    # ============================================================

    def update_pair(self, j, mu, sigma):
        """
        Update một directed pair (ego, j).

        mu:
            ensemble mean influence estimate.

        sigma:
            uncertainty scale. Code nên truyền standard deviation hoặc một
            uncertainty magnitude non-negative. Nếu proxy vẫn trả variance,
            hàm này vẫn chạy nhưng semantic của sigma_bar sẽ là smoothed
            variance-like uncertainty.
        """
        j = int(j)

        if j not in self.mu_bar:
            return

        mu = float(mu)
        sigma = self._safe_sigma(sigma)

        lam = self.lambda_0 / (1.0 + self.uncertainty_scale * sigma)
        lam = float(np.clip(lam, 0.0, 1.0))

        self.mu_bar[j] = (1.0 - lam) * float(self.mu_bar[j]) + lam * mu
        self.sigma_bar[j] = (1.0 - lam) * float(self.sigma_bar[j]) + lam * sigma

        sigma_for_score = self._safe_sigma(self.sigma_bar[j])
        score = (abs(float(self.mu_bar[j])) - self.tau) / (sigma_for_score + self.eps)
        self.p_core[j] = float(self._sigmoid(score))

        self.mu_history[j].append(float(self.mu_bar[j]))
        self.sigma_history[j].append(float(self.sigma_bar[j]))
        self.p_history[j].append(float(self.p_core[j]))

        if len(self.mu_history[j]) > 500:
            self.mu_history[j] = self.mu_history[j][-500:]

        if len(self.sigma_history[j]) > 500:
            self.sigma_history[j] = self.sigma_history[j][-500:]

        if len(self.p_history[j]) > 500:
            self.p_history[j] = self.p_history[j][-500:]

    def update_batch(self, mu_sigma_dict):
        """
        Update nhiều pair rồi cập nhật core set bằng hysteresis.

        Return:
            (promoted, demoted)

        Runner đang cần return này để:
            pair_rel_module.warm_start_if_promoted(ego, promoted)
        """
        for j, pair_value in mu_sigma_dict.items():
            if pair_value is None:
                continue

            mu, sigma = pair_value
            self.update_pair(j, mu, sigma)

        promoted, demoted = self._update_core_set_hysteresis()
        return promoted, demoted

    def _update_core_set_hysteresis(self):
        """
        Learned-belief takeover core update.

        Candidate core trước tiên được chọn bằng hysteresis:
            - p > tau_in thì vào.
            - nếu đã ở core và p >= tau_out thì giữ.

        Sau đó:
            - fill nếu nhỏ hơn min_core_size.
            - prune nếu lớn hơn max_core_size.

        Đây là chỗ sửa trực tiếp lỗi core=21 khi max_core_size=4.
        """
        old_core = set(self.core_set)
        self.prev_core_set = set(self.core_set)

        new_core = set()

        for j in self.neighbor_ids:
            p = float(self.p_core[j])

            if p > self.tau_in:
                new_core.add(j)
            elif (j in self.prev_core_set) and (p >= self.tau_out):
                new_core.add(j)

        new_core = self._apply_core_capacity_constraints(new_core)

        self.core_set = set(new_core)
        self._record_core_change(old_core, self.core_set)

        return set(self.last_promoted), set(self.last_demoted)

    # ============================================================
    # Public accessors used by runners
    # ============================================================

    def get_core_set(self):
        return set(self.core_set)

    def get_peripheral_set(self):
        return set(self.neighbor_ids) - set(self.core_set)

    def get_state_for_neighbor(self, j):
        j = int(j)

        return {
            "mu_bar": float(self.mu_bar[j]),
            "sigma_bar": float(self.sigma_bar[j]),
            "p_core": float(self.p_core[j]),
            "in_core": float(j in self.core_set),
            "in_seed_core": float(j in self.seeded_core_set),
        }

    def get_state_dict(self):
        return {
            int(j): self.get_state_for_neighbor(j)
            for j in self.neighbor_ids
        }

    def get_mean_uncertainty(self):
        vals = [float(self.sigma_bar[j]) for j in self.neighbor_ids]
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    def get_temporal_variance(self, window=50):
        vals = []

        for j in self.neighbor_ids:
            hist = self.mu_history[j][-int(window):]

            if len(hist) > 1:
                vals.append(float(np.var(hist)))

        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    def get_core_switch_count(self):
        return int(self.last_core_switch_count)

    def reset_switch_counter(self):
        """
        Dùng sau khi Stage 0 chuyển sang Stage 1.

        Tránh để Stage 0 re-seeding làm nhiễu metric core switch của learned belief.
        """
        self.last_promoted = set()
        self.last_demoted = set()
        self.last_core_switch_count = 0
        self.prev_core_set = set(self.core_set)

    def get_last_promoted(self):
        return set(self.last_promoted)

    def get_last_demoted(self):
        return set(self.last_demoted)

    # ============================================================
    # Diagnostics
    # ============================================================

    def get_population_debug_stats(self):
        """
        Debug cho một ego.

        Dùng để kiểm tra lỗi kiểu:
            core=21 nhưng max_core_size=4
            max_p thấp nhưng core quá lớn
            min_core_size bị set sai
        """
        mu_vals = [abs(float(self.mu_bar[j])) for j in self.neighbor_ids]
        sigma_vals = [float(self.sigma_bar[j]) for j in self.neighbor_ids]
        p_vals = [float(self.p_core[j]) for j in self.neighbor_ids]

        num_above_tau_in = sum(
            1 for j in self.neighbor_ids
            if float(self.p_core[j]) > self.tau_in
        )

        num_kept_by_tau_out = sum(
            1 for j in self.neighbor_ids
            if (j in self.prev_core_set) and (float(self.p_core[j]) >= self.tau_out)
        )

        return {
            "ego_id": int(self.ego_id),
            "n_neighbors": int(len(self.neighbor_ids)),
            "core_size": int(len(self.core_set)),
            "peripheral_size": int(len(self.get_peripheral_set())),
            "min_core_size": int(self.min_core_size),
            "max_core_size": int(self.max_core_size),
            "tau": float(self.tau),
            "tau_in": float(self.tau_in),
            "tau_out": float(self.tau_out),
            "mean_abs_mu": float(np.mean(mu_vals)) if len(mu_vals) > 0 else 0.0,
            "max_abs_mu": float(np.max(mu_vals)) if len(mu_vals) > 0 else 0.0,
            "mean_sigma": float(np.mean(sigma_vals)) if len(sigma_vals) > 0 else 0.0,
            "min_sigma": float(np.min(sigma_vals)) if len(sigma_vals) > 0 else 0.0,
            "max_sigma": float(np.max(sigma_vals)) if len(sigma_vals) > 0 else 0.0,
            "mean_p_core": float(np.mean(p_vals)) if len(p_vals) > 0 else 0.0,
            "max_p_core": float(np.max(p_vals)) if len(p_vals) > 0 else 0.0,
            "num_above_tau_in": int(num_above_tau_in),
            "num_kept_by_tau_out": int(num_kept_by_tau_out),
            "last_promoted": sorted([int(x) for x in self.last_promoted]),
            "last_demoted": sorted([int(x) for x in self.last_demoted]),
            "last_core_switch_count": int(self.last_core_switch_count),
            "core_set": sorted([int(x) for x in self.core_set]),
        }

    def get_mean_abs_mu(self):
        vals = [abs(float(self.mu_bar[j])) for j in self.neighbor_ids]
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    def get_max_p_core(self):
        vals = [float(self.p_core[j]) for j in self.neighbor_ids]
        return float(np.max(vals)) if len(vals) > 0 else 0.0