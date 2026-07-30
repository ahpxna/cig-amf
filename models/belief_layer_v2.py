"""
belief_layer_v2.py — Bayes-light structural belief, bản vá.

=============================================================================
LỖI CỦA BẢN v1 VÀ CHẨN ĐOÁN BẰNG CHÍNH SỐ LIỆU TRONG PAPER
=============================================================================

[B1] p_core BÃO HOÀ — sigma nằm ở MẪU SỐ.
     v1 (belief_layer.py dòng 281):
         score  = (|mu_bar| - tau) / (sigma_bar + eps)
         p_core = sigmoid(score)

     sigma nhỏ -> mẫu số nhỏ -> đối số sigmoid bị khuếch đại -> BÃO HOÀ 0/1.
     Kiểm chứng bằng bảng kết quả của chính paper:
        NoTwoTimescale: sigma = 0.050 -> chia 0.05 tức nhân 20 -> bão hoà
                        -> chỉ 1 neighbour sống sót -> core = 1.0 (đúng như quan sát)
        Final CIG-AMF:  sigma = 0.663 -> sigmoid mềm -> nhiều neighbour vượt tau_in
                        -> core chạm trần max_core_size (đúng như quan sát 6.0)

     Nghĩa là: CORE SIZE BỊ QUYẾT ĐỊNH BỞI ĐỘ LỚN CỦA SIGMA, không phải bởi
     cấu trúc ảnh hưởng. Tín hiệu mu (thứ mang thông tin nhân quả) bị nhấn chìm.

     Nguyên nhân sâu xa: sigma bị dùng HAI LẦN — một lần điều tiết learning
     rate (Eq. 13, hợp lý) và một lần làm mẫu số ngưỡng (Eq. 14, gây bão hoà).
     Hiệu ứng nhân đôi.

     v2: TÁCH HAI VAI TRÒ. sigma chỉ còn điều tiết tốc độ học. Việc chọn core
     dùng CẬN DƯỚI TIN CẬY (lower confidence bound):
         score_lcb = |mu_bar| - kappa * sigma_bar
     Bất định cao -> bị TRỪ đi -> khó vào core (thận trọng, đúng trực giác),
     nhưng KHÔNG bão hoà vì sigma ở dạng cộng chứ không phải chia.

[B2] CORE SIZE LÀ HẰNG SỐ CỨNG, KHÔNG PHẢI KẾT QUẢ HỌC.
     v1 có min_core_size / max_core_size. Kết quả: 6.0 +- 0.0 và 1.0 +- 0.0
     và 2.0 +- 0.0 trên 5 seed. Std = 0 tuyệt đối cho một đại lượng "được học"
     là điều gần như không thể. Cả ba đều là ĐIỂM BIÊN (chạm trần / sụp đáy /
     đứng nguyên seed), không có giá trị nào là "lựa chọn".

     v2: giữ min/max làm van an toàn NHƯNG:
       - đếm và báo cáo tỷ lệ chạm biên (get_saturation_stats)
       - thêm chế độ adaptive_k: k tự co giãn theo entropy của phân phối
         ảnh hưởng. Ảnh hưởng tập trung vào ít neighbour -> k nhỏ.
         Ảnh hưởng dàn đều -> k lớn. Khi đó k mới thật sự "adaptive".

[B3] alpha KHÔNG THOẢ ROBBINS-MONRO -> không thể phát biểu định lý hội tụ.
     v1: alpha = lambda_0 / (1 + c * sigma)
     Bị chặn dưới bởi lambda_0/(1+c*sigma_max) > 0 -> sum(alpha^2) = vô cùng.
     Pieroth (ICML 2024) Theorem 5.6 chứng minh hội tụ a.s. cho đúng dạng
     iteration này, với điều kiện Assumption 3.3(c):
         sum(alpha_t) = vô cùng   VÀ   sum(alpha_t^2) < vô cùng
     Họ dùng alpha_t = alpha_0 / t^d với d = 0.726.

     v2: alpha_t = lambda_0 / (t^decay * (1 + c * sigma))
     với decay in (0.5, 1]. Giờ thoả Robbins-Monro -> MƯỢN ĐƯỢC khung chứng
     minh của Pieroth -> paper có thêm một mệnh đề hội tụ gần như miễn phí.

[B4] BELIEF KHÔNG CÓ DẤU.
     v1 lưu mu_bar rồi mọi chỗ đều lấy |mu_bar|. Không phân biệt được
     "thằng ngáng đường" với "thằng hỗ trợ".
     v2: giữ dấu xuyên suốt. Đây là đầu vào cho slot ngữ nghĩa Thiện/Ác.
=============================================================================
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class BayesLightBeliefStateV2:
    """
    Structural belief cho một ego-agent i.

    State mỗi directed pair (i, j):
        mu_bar     : ảnh hưởng đã làm mượt, CÓ DẤU
        sigma_bar  : bất định đã làm mượt (epistemic, từ ensemble)
        p_core     : điểm xác suất vào core, KHÔNG còn bão hoà
        n_updates  : số lần đã cập nhật (cho lịch trình Robbins-Monro)

    Args:
        core_rule:
            "lcb"      — mặc định. score = |mu| - kappa*sigma  (khuyến nghị)
            "p_core"   — dùng ngưỡng trên p_core như v1 (để chạy ablation)
            "signed"   — chọn riêng top thằng GIÚP và top thằng HẠI, cân bằng dấu
        kappa:
            hệ số phạt bất định trong LCB. Lớn -> thận trọng hơn, core nhỏ hơn.
        alpha_decay:
            số mũ d trong lịch trình Robbins-Monro alpha ~ 1/t^d.
            Phải thuộc (0.5, 1] để thoả sum(a)=inf, sum(a^2)<inf.
            0 = tắt (quay về hành vi v1, dùng cho ablation).
        adaptive_k:
            nếu True, k mục tiêu tự co giãn theo entropy phân phối ảnh hưởng.
    """

    def __init__(
        self,
        ego_id: int,
        neighbor_ids: List[int],
        lambda_0: float = 0.12,
        uncertainty_scale: float = 2.0,
        tau: float = 0.10,
        tau_in: float = 0.62,
        tau_out: float = 0.46,
        weak_prior_top_k: int = 2,
        min_core_size: int = 1,
        max_core_size: int = 4,
        sigma_floor: float = 0.01,
        eps: float = 1e-6,
        # ---- mới ở v2 ----
        core_rule: str = "lcb",
        kappa: float = 1.0,
        alpha_decay: float = 0.7,
        adaptive_k: bool = False,
        adaptive_k_min: int = 1,
        signed_balance: float = 0.5,
    ):
        self.ego_id = int(ego_id)
        self.neighbor_ids = [int(j) for j in neighbor_ids]

        self.lambda_0 = float(lambda_0)
        self.uncertainty_scale = float(uncertainty_scale)
        self.tau = float(tau)
        self.tau_in = float(tau_in)
        self.tau_out = float(tau_out)

        self.weak_prior_top_k = int(weak_prior_top_k)
        self.min_core_size = max(0, int(min_core_size))
        self.max_core_size = int(max_core_size)

        self.sigma_floor = float(sigma_floor)
        self.eps = float(eps)

        if self.max_core_size <= 0:
            self.max_core_size = len(self.neighbor_ids)

        self.max_core_size = min(self.max_core_size, len(self.neighbor_ids))
        self.min_core_size = min(self.min_core_size, self.max_core_size)

        if core_rule not in ("lcb", "p_core", "signed"):
            raise ValueError(f"core_rule không hợp lệ: {core_rule}")

        self.core_rule = str(core_rule)
        self.kappa = float(kappa)
        self.alpha_decay = float(alpha_decay)
        self.adaptive_k = bool(adaptive_k)
        self.adaptive_k_min = int(max(1, adaptive_k_min))
        self.signed_balance = float(np.clip(signed_balance, 0.0, 1.0))

        # ---- state ----
        self.mu_bar: Dict[int, float] = {j: 0.0 for j in self.neighbor_ids}
        self.sigma_bar: Dict[int, float] = {j: 1.0 for j in self.neighbor_ids}
        self.p_core: Dict[int, float] = {j: 0.5 for j in self.neighbor_ids}
        self.n_updates: Dict[int, int] = {j: 0 for j in self.neighbor_ids}

        # Tích luỹ prod(1-alpha) cho bias correction. Bắt đầu ở 1.0 =
        # "toàn bộ khối lượng còn nằm ở giá trị khởi tạo".
        self._bias_corr: Dict[int, float] = {j: 1.0 for j in self.neighbor_ids}

        # ĐIỂM NEO theo từng cặp. Ban đầu = hằng số lớp, nhưng sau mỗi lần
        # re-anchor (inflate_uncertainty) nó được đặt lại thành ước lượng
        # tốt nhất đang có. Nhờ vậy việc "quên có kiểm soát" khi phát hiện
        # structural shift không xoá trắng những gì đã học.
        self._mu_init: Dict[int, float] = {
            j: self.MU_INIT for j in self.neighbor_ids
        }
        self._sigma_init: Dict[int, float] = {
            j: self.SIGMA_INIT for j in self.neighbor_ids
        }

        # Đếm số lần bơm phồng, để báo cáo trong paper.
        self.n_inflations = 0
        self.last_inflation_episode = None

        self.core_set: Set[int] = set()
        self.prev_core_set: Set[int] = set()
        self.seeded_core_set: Set[int] = set()

        self.last_promoted: Set[int] = set()
        self.last_demoted: Set[int] = set()
        self.last_core_switch_count = 0

        self.mu_history: Dict[int, List[float]] = {j: [] for j in self.neighbor_ids}
        self.sigma_history: Dict[int, List[float]] = {j: [] for j in self.neighbor_ids}
        self.p_history: Dict[int, List[float]] = {j: [] for j in self.neighbor_ids}
        self.core_history: List[Set[int]] = []

        # [B2] Đếm chạm biên — con số PHẢI báo cáo trong paper.
        self.n_core_updates = 0
        self.n_hit_max = 0
        self.n_hit_min = 0

    # =====================================================================
    # Helper
    # =====================================================================

    # Giá trị khởi tạo của belief — cần cho bias correction tổng quát.
    MU_INIT = 0.0
    SIGMA_INIT = 1.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    def _safe_sigma(self, sigma) -> float:
        return max(float(sigma), self.sigma_floor, self.eps)

    def debiased_mu(self, j: int) -> float:
        """
        mu_bar đã khử chệch khởi tạo.

            mu_hat = mu_bar / (1 - prod(1 - alpha_s))

        Dùng cái này ở MỌI CHỖ RA QUYẾT ĐỊNH (chọn core, gán vai trò),
        còn mu_bar thô chỉ để log.
        """
        j = int(j)
        prod = float(self._bias_corr.get(j, 1.0))
        denom = 1.0 - prod

        mu_init = float(self._mu_init.get(j, self.MU_INIT))

        if denom < 1e-3:
            # Chưa tích luỹ đủ bằng chứng vượt lên trên điểm neo.
            # Trả về CHÍNH ĐIỂM NEO (không phải 0.0 cứng như bản trước):
            #  - lúc khởi động, điểm neo = 0.0 -> hành vi y hệt bản cũ
            #  - sau khi re-anchor (inflate_uncertainty), điểm neo là ước
            #    lượng tốt nhất đang có -> không bị xoá trắng về 0
            return mu_init

        return (float(self.mu_bar[j]) - prod * mu_init) / denom

    def debiased_sigma(self, j: int) -> float:
        """
        sigma_bar đã khử chệch. Không có bước này, prior sigma=1.0 còn sót
        lại làm sigma bị THỔI PHỒNG, khiến LCB âm và core rỗng.
        """
        j = int(j)
        prod = float(self._bias_corr.get(j, 1.0))
        denom = 1.0 - prod

        sigma_init = float(self._sigma_init.get(j, self.SIGMA_INIT))

        if denom < 1e-3:
            return max(sigma_init, self.sigma_floor)

        # ---------------------------------------------------------------
        # CÔNG THỨC TỔNG QUÁT — lỗi này bị bắt trong unit test.
        #
        # Adam dùng m/(1-beta^t) vì m KHỞI TẠO BẰNG 0. Với sigma ta khởi tạo
        # bằng 1.0 (prior "bất định tối đa"), nên:
        #     sigma_bar = prod*1.0 + (1-prod)*sigma_thuc
        # Chia thẳng cho (1-prod) sẽ cho  prod/(1-prod) + sigma_thuc,
        # tức THỔI PHỒNG sigma. Số đo được từ test: sigma_deb = 1.15 trong
        # khi giá trị thật ~0.19 (gấp 6 lần!). Hệ quả: LCB = |mu| - kappa*sigma
        # luôn âm -> KHÔNG AI vào core -> core rơi về min_core_size.
        #
        # Đúng phải TRỪ phần đóng góp còn sót của giá trị khởi tạo trước:
        #     sigma_hat = (sigma_bar - prod*init) / (1 - prod)
        # ---------------------------------------------------------------
        val = (float(self.sigma_bar[j]) - prod * sigma_init) / denom

        return max(val, self.sigma_floor)

    def _lcb_score(self, j: int) -> float:
        """
        [B1] Cận dưới tin cậy — thay cho công thức bão hoà của v1.

            score = |mu_bar| - kappa * sigma_bar

        Trực giác: "tôi tin chắc ít nhất bằng ngần này". Bất định cao bị TRỪ
        đi chứ không CHIA vào, nên không bao giờ bão hoà.
        Đây là chuẩn mực trong bandit (LCB/UCB) và khớp tinh thần CASEC
        (dùng phương sai payoff để thưa hoá cạnh).
        """
        j = int(j)
        mu = self.debiased_mu(j)
        sigma = self._safe_sigma(self.debiased_sigma(j))

        return float(abs(mu) - self.kappa * sigma)

    def _priority(self, j: int) -> float:
        """Ưu tiên khi fill/prune. Dùng cùng thang với luật chọn core."""
        if self.core_rule == "p_core":
            j = int(j)
            mu = abs(float(self.mu_bar.get(j, 0.0)))
            sigma = self._safe_sigma(self.sigma_bar.get(j, 1.0))
            return float(mu / (sigma + self.eps))

        return self._lcb_score(j)

    def _rank(self, candidates=None) -> List[int]:
        if candidates is None:
            candidates = self.neighbor_ids

        cands = [int(j) for j in candidates if int(j) in self.mu_bar]

        return sorted(cands, key=lambda j: self._priority(j), reverse=True)

    # =====================================================================
    # [B2] k thích nghi
    # =====================================================================

    def _effective_max_k(self) -> int:
        """
        Nếu adaptive_k bật, k mục tiêu co giãn theo ENTROPY của phân phối
        ảnh hưởng đã chuẩn hoá.

        Trực giác:
          - Ảnh hưởng tập trung vào 1-2 neighbour (entropy thấp) -> k nhỏ,
            không cần nhiều core slot.
          - Ảnh hưởng dàn đều (entropy cao) -> k lớn hơn, vì không có ai
            nổi trội rõ ràng.

        Đây là chỗ biến "adaptive" từ tên gọi thành cơ chế thật.
        """
        if not self.adaptive_k:
            return int(self.max_core_size)

        vals = np.array(
            [abs(float(self.mu_bar[j])) for j in self.neighbor_ids],
            dtype=np.float64,
        )  # [n_neighbors]

        total = float(np.sum(vals))

        if total <= self.eps or len(vals) <= 1:
            return int(self.max_core_size)

        p = vals / total                                  # [n_neighbors]
        p = np.clip(p, 1e-12, 1.0)

        entropy = float(-np.sum(p * np.log(p)))
        max_entropy = float(np.log(len(vals)))

        # frac in [0,1]: 0 = cực tập trung, 1 = dàn đều hoàn toàn
        frac = entropy / max(max_entropy, 1e-12)

        k = int(round(
            self.adaptive_k_min
            + frac * (self.max_core_size - self.adaptive_k_min)
        ))

        return int(np.clip(k, self.adaptive_k_min, self.max_core_size))

    def _apply_capacity(self, candidate_core: Set[int]) -> Set[int]:
        core = set(int(j) for j in candidate_core if int(j) in self.mu_bar)

        eff_max = self._effective_max_k()

        self.n_core_updates += 1

        if len(core) < self.min_core_size:
            self.n_hit_min += 1

            for j in self._rank(self.neighbor_ids):
                core.add(int(j))
                if len(core) >= self.min_core_size:
                    break

        if len(core) > eff_max:
            self.n_hit_max += 1
            core = set(self._rank(core)[:eff_max])

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

    # =====================================================================
    # Stage 0 seeding
    # =====================================================================

    def initialize_from_weak_prior(self, prior_scores: Dict[int, float]):
        """Giữ nguyên hành vi v1."""
        ranked = sorted(
            [(int(j), float(prior_scores.get(j, 0.0))) for j in self.neighbor_ids],
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
            self.p_core[j] = 0.58 if j in self.seeded_core_set else 0.42

        self._record_core_change(old_core, self.core_set)

    def set_fixed_core(self, core_ids):
        old_core = set(self.core_set)

        cand = set(
            int(j) for j in core_ids if int(j) in self.neighbor_ids
        )
        cand = self._apply_capacity(cand)

        self.core_set = set(cand)
        self.prev_core_set = set(cand)
        self.seeded_core_set = set(cand)

        for j in self.neighbor_ids:
            self.p_core[j] = 0.58 if j in self.core_set else 0.42

        self._record_core_change(old_core, self.core_set)

    # =====================================================================
    # Cập nhật belief
    # =====================================================================

    def update_pair(self, j: int, mu: float, sigma: float):
        """
        Cập nhật một cặp.

        mu: CÓ DẤU (khác v1 — v1 nhận mu không âm vì proxy đã lấy abs).
        sigma: standard deviation qua ensemble.
        """
        j = int(j)

        if j not in self.mu_bar:
            return

        mu = float(mu)
        sigma = self._safe_sigma(sigma)

        self.n_updates[j] += 1
        t = float(self.n_updates[j])

        # ---------------------------------------------------------------
        # [B3] LỊCH TRÌNH ROBBINS-MONRO
        #
        #   alpha_t = lambda_0 / ( t^d * (1 + c * sigma_t) )
        #
        # Phần t^-d cho:  sum(alpha) = inf,  sum(alpha^2) < inf  khi d in (0.5, 1]
        #   -> thoả Assumption 3.3(c) của Pieroth ICML'24
        #   -> mượn được Theorem 5.6 để phát biểu hội tụ almost surely
        # Phần (1 + c*sigma) giữ ý tưởng gốc của paper: bất định cao thì
        #   cập nhật dè dặt hơn.
        # alpha_decay = 0 -> quay về đúng công thức v1 (dùng cho ablation).
        # ---------------------------------------------------------------
        decay_factor = t ** self.alpha_decay if self.alpha_decay > 0.0 else 1.0

        alpha = self.lambda_0 / (
            decay_factor * (1.0 + self.uncertainty_scale * sigma)
        )
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # [B4] Giữ DẤU của mu.
        self.mu_bar[j] = (1.0 - alpha) * float(self.mu_bar[j]) + alpha * mu
        self.sigma_bar[j] = (
            (1.0 - alpha) * float(self.sigma_bar[j]) + alpha * sigma
        )

        # ---------------------------------------------------------------
        # HIỆU CHỈNH ĐỘ CHỆCH KHỞI TẠO (bias correction, kiểu Adam)
        #
        # Lỗi này bị bắt trong unit test và nó là HỆ QUẢ PHỤ của [B3].
        # Khi alpha suy giảm theo 1/t^d, tổng trọng số đã tích luỹ được là
        #     1 - prod_s(1 - alpha_s)
        # và phần còn lại VẪN THUỘC VỀ GIÁ TRỊ KHỞI TẠO.
        #
        # Con số cụ thể từ test: sau 60 lần cập nhật với lambda_0=0.12,
        # d=0.7, thì prod(1-alpha) ~ 0.38. Nghĩa là:
        #     mu_bar   chỉ đạt ~62% giá trị thật (bị kéo về 0)
        #     sigma_bar vẫn giữ ~38% của prior 1.0 -> bị THỔI PHỒNG
        # Kết hợp lại: LCB = |mu| - kappa*sigma luôn ÂM -> KHÔNG AI vào core
        # -> core rơi về min_core_size = 1. Đúng như test đã cho thấy.
        #
        # Adam giải đúng bài toán này bằng m_hat = m / (1 - beta^t).
        # Ở đây alpha thay đổi theo bước nên ta tích luỹ tích số trực tiếp.
        # ---------------------------------------------------------------
        self._bias_corr[j] = float(self._bias_corr[j]) * (1.0 - alpha)

        # ---------------------------------------------------------------
        # [B1] p_core KHÔNG CÒN CHIA CHO SIGMA.
        # Dạng mới: sigmoid trên LCB đã chuẩn hoá bằng một hằng số CỐ ĐỊNH
        # (không phụ thuộc sigma) -> không bao giờ bão hoà theo sigma.
        # p_core giờ chỉ còn là một số để log/diagnostics; quyết định core
        # thật sự nằm ở _select_core_lcb().
        # ---------------------------------------------------------------
        lcb = self._lcb_score(j)
        self.p_core[j] = float(self._sigmoid((lcb - self.tau) / max(self.tau, 1e-3)))

        for hist, val in (
            (self.mu_history[j], self.mu_bar[j]),
            (self.sigma_history[j], self.sigma_bar[j]),
            (self.p_history[j], self.p_core[j]),
        ):
            hist.append(float(val))
            if len(hist) > 500:
                del hist[:-500]

    def update_batch(self, mu_sigma_dict) -> Tuple[Set[int], Set[int]]:
        """
        GIỮ NGUYÊN chữ ký v1: nhận {j: (mu, sigma)}, trả (promoted, demoted).
        """
        for j, pair_value in mu_sigma_dict.items():
            if pair_value is None:
                continue

            mu, sigma = pair_value
            self.update_pair(j, mu, sigma)

        return self._update_core_set()

    # =====================================================================
    # Chọn core
    # =====================================================================

    def _select_core_lcb(self) -> Set[int]:
        """
        Luật mặc định v2.

        Vào core nếu:   lcb_score > tau
        Giữ trong core nếu: lcb_score > tau_hold  (tau_hold < tau, hysteresis)

        Hysteresis giữ nguyên tinh thần v1 (chống nhấp nháy) nhưng áp lên
        thang LCB thay vì thang p_core bão hoà.
        """
        # Quy đổi tỷ lệ tau_out/tau_in của v1 sang thang LCB.
        ratio = (
            self.tau_out / self.tau_in if self.tau_in > 1e-8 else 0.75
        )
        tau_hold = self.tau * float(np.clip(ratio, 0.0, 1.0))

        new_core = set()

        for j in self.neighbor_ids:
            score = self._lcb_score(j)

            if score > self.tau:
                new_core.add(j)
            elif (j in self.prev_core_set) and (score > tau_hold):
                new_core.add(j)

        return new_core

    def _select_core_signed(self) -> Set[int]:
        """
        Luật CÓ DẤU — chọn cân bằng giữa thằng GIÚP và thằng HẠI.

        Vì sao đáng thử: MAGIC (2026) chỉ ra "ảnh hưởng mạnh chưa chắc có ích".
        Nếu chỉ xếp hạng theo |mu|, core có thể bị chiếm hết bởi một loại.
        Ở đây ta dành signed_balance phần slot cho phía hại (thường quan trọng
        hơn để né) và phần còn lại cho phía giúp.

        Đây là biến thể chưa thấy ai làm — MAGIC dùng dấu để LỌC REWARD,
        còn đây dùng dấu để CÂN BẰNG NGÂN SÁCH CORE.
        """
        eff_max = self._effective_max_k()

        n_harm = int(round(self.signed_balance * eff_max))
        n_help = eff_max - n_harm

        harmful = [
            j for j in self.neighbor_ids
            if float(self.mu_bar[j]) < 0.0 and self._lcb_score(j) > self.tau
        ]
        helpful = [
            j for j in self.neighbor_ids
            if float(self.mu_bar[j]) >= 0.0 and self._lcb_score(j) > self.tau
        ]

        harmful.sort(key=lambda j: self._lcb_score(j), reverse=True)
        helpful.sort(key=lambda j: self._lcb_score(j), reverse=True)

        new_core = set(harmful[:n_harm]) | set(helpful[:n_help])

        # Nếu một phía thiếu, bù bằng phía kia (không lãng phí slot).
        if len(new_core) < eff_max:
            leftovers = [
                j for j in self._rank(self.neighbor_ids)
                if j not in new_core and self._lcb_score(j) > self.tau
            ]
            for j in leftovers:
                new_core.add(j)
                if len(new_core) >= eff_max:
                    break

        return new_core

    def _select_core_p(self) -> Set[int]:
        """Luật cũ của v1 — giữ để chạy ablation 'trước khi vá'."""
        new_core = set()

        for j in self.neighbor_ids:
            p = float(self.p_core[j])

            if p > self.tau_in:
                new_core.add(j)
            elif (j in self.prev_core_set) and (p >= self.tau_out):
                new_core.add(j)

        return new_core

    def _update_core_set(self) -> Tuple[Set[int], Set[int]]:
        old_core = set(self.core_set)
        self.prev_core_set = set(self.core_set)

        if self.core_rule == "lcb":
            new_core = self._select_core_lcb()
        elif self.core_rule == "signed":
            new_core = self._select_core_signed()
        else:
            new_core = self._select_core_p()

        new_core = self._apply_capacity(new_core)

        self.core_set = set(new_core)
        self._record_core_change(old_core, self.core_set)

        return set(self.last_promoted), set(self.last_demoted)

    # =====================================================================
    # Accessors (giữ nguyên API v1)
    # =====================================================================

    def get_core_set(self) -> Set[int]:
        return set(self.core_set)

    def get_peripheral_set(self) -> Set[int]:
        return set(self.neighbor_ids) - set(self.core_set)

    def get_state_for_neighbor(self, j: int) -> Dict[str, float]:
        j = int(j)

        return {
            # Xuất bản ĐÃ KHỬ CHỆCH — đây là thứ peripheral memory dùng
            # để gán slot ngữ nghĩa, nên phải là ước lượng đúng thang.
            "mu_bar": float(self.debiased_mu(j)),      # CÓ DẤU
            "sigma_bar": float(self.debiased_sigma(j)),
            "mu_bar_raw": float(self.mu_bar[j]),
            "p_core": float(self.p_core[j]),
            "in_core": float(j in self.core_set),
            "in_seed_core": float(j in self.seeded_core_set),
            # mới: để peripheral memory dùng trực tiếp
            "lcb_score": float(self._lcb_score(j)),
            "n_updates": int(self.n_updates[j]),
        }

    def get_state_dict(self) -> Dict[int, Dict[str, float]]:
        return {int(j): self.get_state_for_neighbor(j) for j in self.neighbor_ids}

    def get_mean_uncertainty(self) -> float:
        vals = [float(self.sigma_bar[j]) for j in self.neighbor_ids]
        return float(np.mean(vals)) if vals else 0.0

    def get_temporal_variance(self, window: int = 50) -> float:
        """
        LƯU Ý CHO PAPER: v1 báo cáo đại lượng này ~ 1e-8 và diễn giải là
        "ổn định". Nhưng 1e-8 về bản chất là BẰNG KHÔNG — belief không nhúc
        nhích, tức ĐÓNG BĂNG chứ không phải ổn định. Hai thứ khác nhau.
        Nên báo cáo kèm normalised_temporal_variance bên dưới.
        """
        vals = []

        for j in self.neighbor_ids:
            hist = self.mu_history[j][-int(window):]
            if len(hist) > 1:
                vals.append(float(np.var(hist)))

        return float(np.mean(vals)) if vals else 0.0

    def get_normalised_temporal_variance(self, window: int = 50) -> float:
        """
        Phương sai theo thời gian CHUẨN HOÁ theo thang của chính mu.

            nvar = var(mu_t) / (mean(|mu_t|)^2 + eps)

        Đại lượng không thứ nguyên -> so sánh được giữa các phương pháp,
        và không bị "nhỏ giả tạo" chỉ vì mu bé.
        """
        vals = []

        for j in self.neighbor_ids:
            hist = np.asarray(self.mu_history[j][-int(window):], dtype=np.float64)

            if hist.shape[0] > 1:
                scale = float(np.mean(np.abs(hist))) ** 2
                vals.append(float(np.var(hist)) / (scale + 1e-12))

        return float(np.mean(vals)) if vals else 0.0

    def get_core_switch_count(self) -> int:
        return int(self.last_core_switch_count)

    def reset_switch_counter(self):
        self.last_promoted = set()
        self.last_demoted = set()
        self.last_core_switch_count = 0
        self.prev_core_set = set(self.core_set)

    def get_last_promoted(self) -> Set[int]:
        return set(self.last_promoted)

    def get_last_demoted(self) -> Set[int]:
        return set(self.last_demoted)

    def get_mean_abs_mu(self) -> float:
        vals = [abs(float(self.mu_bar[j])) for j in self.neighbor_ids]
        return float(np.mean(vals)) if vals else 0.0

    def get_max_p_core(self) -> float:
        vals = [float(self.p_core[j]) for j in self.neighbor_ids]
        return float(np.max(vals)) if vals else 0.0

    # =====================================================================
    # BƠM PHỒNG BẤT ĐỊNH — "quên có kiểm soát" khi phát hiện structural shift
    # =====================================================================

    def inflate_uncertainty(
        self,
        factor: float = 2.5,
        t_reset: int = 1,
        pairs: Optional[List[int]] = None,
        sigma_ceiling: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Gọi khi residual/matrix trigger báo có structural shift.

        VẤN ĐỀ NÓ GIẢI QUYẾT
        --------------------
        Lịch trình Robbins-Monro alpha ~ 1/t^d là điều kiện để có định lý hội
        tụ, nhưng nó tạo một nghịch lý: càng học lâu, alpha càng nhỏ, hệ càng
        "cứng đầu". Sau vài trăm bước, một bằng chứng mới gần như không lay
        chuyển được belief nữa. Trong môi trường DỪNG thì đó là tính năng.
        Trong môi trường PHI DỪNG — đúng bài toán của chúng ta — đó là lỗi:
        cấu trúc đã đổi thật mà hệ không chịu tin.

        Ví dụ đời thường: bạn quen một người 10 năm, đánh giá đã ổn định. Rồi
        họ đổi việc, đổi hoàn cảnh, tính nết khác hẳn. Nếu bạn vẫn cập nhật
        ấn tượng với "trọng số 10 năm" thì phải mất thêm nhiều năm mới nhận
        ra. Người tỉnh táo sẽ nói: "hoàn cảnh nó khác rồi, mình phải xem lại
        từ đầu" — tức TỰ HẠ ĐỘ TIN CẬY của mình xuống và mở lại tốc độ học.

        CƠ CHẾ (hai bước, phải làm cùng nhau)
        -------------------------------------
        1. RE-ANCHOR: lấy ước lượng đã khử chệch hiện tại làm ĐIỂM NEO mới.
           Đây là chỗ khác biệt với "reset về 0": ta không xoá những gì đã
           học, ta chỉ hạ độ tin cậy vào nó.
        2. BƠM PHỒNG + MỞ LẠI TỐC ĐỘ HỌC: nhân sigma với `factor` và đặt lại
           bộ đếm t về `t_reset`, khiến alpha bật lại lên mức cao.

        HIỆU ỨNG DÂY CHUYỀN (đây là chỗ đẹp)
        ------------------------------------
        sigma tăng -> LCB = |mu| - kappa*sigma giảm -> ít neighbour vượt tau
        -> CORE TỰ CO LẠI. Hệ tự động thận trọng trong lúc chưa hiểu tình
        hình mới, rồi tự nở ra khi đã học lại. Không cần luật thủ công nào.

        Đây chính là `covariance inflation` trong bộ lọc Kalman, và cùng họ
        với discounted/sliding-window UCB cho bandit phi dừng.

        Args:
            factor: hệ số nhân sigma. 2.0-3.0 là hợp lý.
            t_reset: đặt lại bộ đếm về giá trị này (1 = mở hết tốc độ học).
            pairs: chỉ áp cho các cặp này. None = toàn bộ.
            sigma_ceiling: trần cho sigma sau khi bơm. None = SIGMA_INIT.

        Returns:
            dict thống kê để log.
        """
        ids = self.neighbor_ids if pairs is None else [
            int(j) for j in pairs if int(j) in self.mu_bar
        ]

        ceiling = (
            float(self.SIGMA_INIT) if sigma_ceiling is None
            else float(sigma_ceiling)
        )

        before = float(np.mean(
            [self.debiased_sigma(j) for j in ids]
        )) if ids else 0.0

        for j in ids:
            # --- bước 1: re-anchor vào ước lượng tốt nhất đang có ---
            mu_hat = self.debiased_mu(j)
            sig_hat = self.debiased_sigma(j)

            # --- bước 2: bơm phồng, có trần ---
            new_sigma = min(
                max(sig_hat * float(factor), self.sigma_floor),
                ceiling,
            )

            self.mu_bar[j] = mu_hat
            self.sigma_bar[j] = new_sigma

            # Điểm neo mới = trạng thái vừa đặt. Kết hợp với _bias_corr=1.0,
            # debiased_* sẽ trả đúng các giá trị này cho tới khi có bằng
            # chứng mới tích luỹ đủ.
            self._mu_init[j] = mu_hat
            self._sigma_init[j] = new_sigma
            self._bias_corr[j] = 1.0

            # Mở lại tốc độ học.
            self.n_updates[j] = int(max(0, t_reset))

        self.n_inflations += 1

        after = float(np.mean(
            [self.debiased_sigma(j) for j in ids]
        )) if ids else 0.0

        return {
            "n_pairs_inflated": int(len(ids)),
            "sigma_before": before,
            "sigma_after": after,
            "factor": float(factor),
            "n_inflations_total": int(self.n_inflations),
        }

    # =====================================================================
    # Chẩn đoán mới
    # =====================================================================

    def get_saturation_stats(self) -> Dict[str, float]:
        """
        [B2] BẮT BUỘC báo cáo trong paper.

        Nếu hit_max_rate ~ 1.0 thì core size KHÔNG phải kết quả học mà là
        trần cứng — và mọi phát biểu về "adaptive capacity allocation" đều
        không có căn cứ. Thà tự nêu ra còn hơn để reviewer tìm thấy.
        """
        n = max(1, self.n_core_updates)

        return {
            "n_core_updates": int(self.n_core_updates),
            "hit_max_rate": float(self.n_hit_max) / float(n),
            "hit_min_rate": float(self.n_hit_min) / float(n),
            "current_core_size": int(len(self.core_set)),
            "effective_max_k": int(self._effective_max_k()),
            "hard_max_core_size": int(self.max_core_size),
        }

    def get_signed_stats(self) -> Dict[str, float]:
        """[B4] Thống kê theo dấu — đầu vào cho slot ngữ nghĩa."""
        mus = np.array(
            [float(self.mu_bar[j]) for j in self.neighbor_ids], dtype=np.float64
        )

        if mus.size == 0:
            return {
                "n_helpful": 0, "n_harmful": 0, "n_neutral": 0,
                "mean_signed_mu": 0.0, "helpful_harmful_ratio": 0.0,
            }

        helpful = int(np.sum(mus > self.tau))
        harmful = int(np.sum(mus < -self.tau))
        neutral = int(mus.size - helpful - harmful)

        return {
            "n_helpful": helpful,
            "n_harmful": harmful,
            "n_neutral": neutral,
            "mean_signed_mu": float(np.mean(mus)),
            "helpful_harmful_ratio": float(helpful) / float(max(1, harmful)),
        }

    def get_population_debug_stats(self) -> Dict:
        mu_abs = [abs(float(self.mu_bar[j])) for j in self.neighbor_ids]
        sig = [float(self.sigma_bar[j]) for j in self.neighbor_ids]
        lcb = [self._lcb_score(j) for j in self.neighbor_ids]

        out = {
            "ego_id": int(self.ego_id),
            "n_neighbors": int(len(self.neighbor_ids)),
            "core_size": int(len(self.core_set)),
            "peripheral_size": int(len(self.get_peripheral_set())),
            "core_rule": self.core_rule,
            "kappa": float(self.kappa),
            "tau": float(self.tau),
            "mean_abs_mu": float(np.mean(mu_abs)) if mu_abs else 0.0,
            "max_abs_mu": float(np.max(mu_abs)) if mu_abs else 0.0,
            "mean_sigma": float(np.mean(sig)) if sig else 0.0,
            "min_sigma": float(np.min(sig)) if sig else 0.0,
            "max_sigma": float(np.max(sig)) if sig else 0.0,
            "mean_lcb": float(np.mean(lcb)) if lcb else 0.0,
            "max_lcb": float(np.max(lcb)) if lcb else 0.0,
            "n_above_tau": int(sum(1 for s in lcb if s > self.tau)),
            "last_core_switch_count": int(self.last_core_switch_count),
            "core_set": sorted(int(x) for x in self.core_set),
        }

        out.update(self.get_saturation_stats())
        out.update(self.get_signed_stats())

        return out


# Alias tương thích ngược.
BayesLightBeliefState = BayesLightBeliefStateV2
