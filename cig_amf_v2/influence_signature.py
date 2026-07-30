"""
influence_signature.py — CHỮ KÝ ẢNH HƯỞNG NHÂN QUẢ. Đây là novelty chính.

=============================================================================
Ý TƯỞNG MỘT CÂU
=============================================================================
Đừng nén ảnh hưởng của một neighbour thành MỘT con số. Giữ nó thành một
HỒ SƠ NHIỀU CHIỀU, rồi dùng hồ sơ đó để gán vai trò chức năng ego-centric.

=============================================================================
VÌ SAO MỘT CON SỐ LÀ KHÔNG ĐỦ
=============================================================================
Hai neighbour có thể có cùng |w| trung bình nhưng khác nhau hoàn toàn:

  Xe A — "blocker" (kẻ chặn):
      dấu: ÂM mạnh | điều kiện: chỉ khi ego tiến vào lane | trễ: TỨC THÌ
      -> w = -0.8 khi ego cách lane 2 ô, w ~ 0 khi ego ở zone khác

  Xe B — "relay/signaller" (kẻ truyền tin):
      dấu: DƯƠNG | điều kiện: rộng | trễ: 3-5 bước
      -> tín hiệu hôm nay giúp ego né kẹt vài bước sau

  Xe C — "consumer" (tranh tài nguyên):
      dấu: âm NHẸ | đều đều mọi ngữ cảnh | tức thì

Trung bình |w| của A và C có thể bằng nhau. Nhưng ego phải đối xử KHÁC NHAU
với chúng. Đây chính là "heterogeneity mà một cái mean không giữ nổi":
mean chỉ giữ trung bình, vứt hết dấu-theo-ngữ-cảnh, độ trễ, độ biến động.

=============================================================================
SÁU CHIỀU CỦA SIGNATURE
=============================================================================
  0. signed_mu       — dấu + độ lớn của ảnh hưởng (giúp hay hại)
  1. abs_mu          — độ lớn thuần (dùng cho magnitude/core selection)
  2. sigma           — bất định epistemic (từ ensemble disagreement)
  3. temporal_std    — biến động theo THỜI GIAN (hành vi j có ổn định không)
  4. context_std     — biến động theo NGỮ CẢNH (ảnh hưởng có điều kiện không)
  5. latency         — trọng tâm horizon của |effect|, trong [0, H-1]

Chiều 3 và 4 khác nhau về bản chất và đây là chỗ dễ nhầm:
  - temporal_std cao = "hôm nay nó ảnh hưởng mạnh, mai lại yếu" -> KHÔNG ỔN ĐỊNH
  - context_std cao  = "ở lane thì ảnh hưởng mạnh, ở zone khác thì không"
                       -> ẢNH HƯỞNG CÓ ĐIỀU KIỆN (đây là tính chất của blocker,
                          và là thứ SCIC/CAI gọi là situation-dependent influence)

=============================================================================
PHÂN ĐỊNH VỚI CÔNG TRÌNH GẦN NHẤT (viết sẵn để dán vào Related Work)
=============================================================================
  ROMA/RODE/SIRD/LDSA/ACORM (role-based MARL):
      role suy từ QUAN SÁT (quỹ đạo, hành vi, tác động lên môi trường)
      -> tương quan; và role là TOÀN CỤC; và dùng để điều kiện hoá POLICY
         CỦA CHÍNH AGENT ĐÓ.
      Điểm chết của họ: hai xe cùng đứng yên — một đứa chặn đường bạn, một
      đứa đậu vô hại — ROMA nhìn hành vi sẽ gán CÙNG role. Chữ ký counterfactual
      tách được ngay vì nó hỏi "nếu nó làm khác thì tôi có khác không".

  Jaques / SCIC / MAGIC (causal influence MARL):
      cũng dùng can thiệp, nhưng biến influence thành INTRINSIC REWARD
      (trả lời "tôi nên LÀM gì"), không dùng để cấu trúc hoá biểu diễn.

  Pieroth ICML'24 (TIM/SIM):
      đo influence structure nhưng CHỦ ĐỘNG TRÁNH counterfactual action,
      đại lượng KHÔNG DẤU (max-min), mục đích MÔ TẢ. Họ tự nêu trong Final
      Remarks rằng "dùng TIM/SIM để cải thiện quá trình học" là FUTURE WORK.

  Ô trống của ta = (tín hiệu interventional CÓ DẤU, NHIỀU CHIỀU, EGO-CENTRIC)
                 x (dùng để TỔ CHỨC BỘ NHỚ + PHÂN BỔ DUNG LƯỢNG)
=============================================================================
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


# Số chiều của signature
SIGNATURE_DIM = 6

# Tên chiều — dùng khi vẽ heatmap centroid cho paper
SIGNATURE_NAMES = (
    "signed_mu",
    "abs_mu",
    "sigma",
    "temporal_std",
    "context_std",
    "latency",
)

# Nhãn vai trò
ROLE_BENEFICIAL = 0   # "Thiện"    — ảnh hưởng dương, mạnh
ROLE_HARMFUL = 1      # "Ác"       — ảnh hưởng âm, mạnh
ROLE_NEUTRAL = 2      # "Trung tính" — quanh mức 0
ROLE_ANOMALOUS = 3    # "Dị biệt"  — bất định cao, chưa hiểu được

ROLE_NAMES = ("beneficial", "harmful", "neutral", "anomalous")
ROLE_NAMES_VI = ("Thiện", "Ác", "Trung tính", "Dị biệt")

N_SEMANTIC_ROLES = 4


class InfluenceSignatureTracker:
    """
    Theo dõi hồ sơ ảnh hưởng nhiều chiều cho MỌI cặp có hướng (i, j).

    Cách dùng trong runner, ngay sau khi gọi proxy.score_batch_full():

        tracker.update(
            ego_id=ego,
            neighbor_id=j,
            signed_mu=out["mu"][b],
            sigma=out["sigma"][b],
            latency=out["latency"][b],
            context_key=env.agent_zone[ego],   # hoặc bất kỳ id ngữ cảnh nào
        )

    Rồi khi cần:
        sig = tracker.get_signature(ego, j)          # np [6]
        role = tracker.get_role(ego, j)              # int 0..3
        mat = tracker.get_signature_matrix(ego, ids) # np [n_ids, 6]

    Args:
        window:
            số quan sát gần nhất giữ lại để tính temporal_std.
        tau_role:
            ngưỡng |signed_mu| để tách Thiện/Ác khỏi Trung tính.
        sigma_hi:
            ngưỡng sigma để gán Dị biệt. Nên đặt theo phân vị của sigma
            quan sát được, không đặt cứng -> dùng auto_calibrate().
        normalise:
            nếu True, get_signature_matrix trả bản đã chuẩn hoá z-score
            theo từng chiều (cần cho k-means, vì các chiều khác thang đo).
    """

    def __init__(
        self,
        n_agents: int,
        window: int = 30,
        tau_role: float = 0.05,
        sigma_hi: float = 0.5,
        normalise: bool = True,
        eps: float = 1e-8,
    ):
        self.n_agents = int(n_agents)
        self.window = int(window)
        self.tau_role = float(tau_role)
        self.sigma_hi = float(sigma_hi)
        self.normalise = bool(normalise)
        self.eps = float(eps)

        # history[(i,j)] = deque các quan sát signed_mu gần nhất
        self._mu_hist: Dict[Tuple[int, int], deque] = {}
        self._sigma_hist: Dict[Tuple[int, int], deque] = {}
        self._latency_hist: Dict[Tuple[int, int], deque] = {}

        # context_mu[(i,j)][ctx] = deque các mu quan sát trong ngữ cảnh ctx
        self._context_mu: Dict[Tuple[int, int], Dict] = {}

        self._n_obs: Dict[Tuple[int, int], int] = {}

    # =====================================================================
    # Cập nhật
    # =====================================================================

    def update(
        self,
        ego_id: int,
        neighbor_id: int,
        signed_mu: float,
        sigma: float,
        latency: float = 0.0,
        context_key=None,
    ):
        """
        Ghi nhận một quan sát ảnh hưởng cho cặp (ego, neighbor).

        context_key:
            định danh ngữ cảnh. Bất cứ thứ gì hashable: zone id, một hash thô
            của vị trí tương đối, hay bin khoảng cách. Đây là thứ cho ta
            chiều context_std — tính điều kiện theo tình huống.
            Nếu None, chiều context_std sẽ bằng 0.
        """
        key = (int(ego_id), int(neighbor_id))

        if key not in self._mu_hist:
            self._mu_hist[key] = deque(maxlen=self.window)
            self._sigma_hist[key] = deque(maxlen=self.window)
            self._latency_hist[key] = deque(maxlen=self.window)
            self._context_mu[key] = {}
            self._n_obs[key] = 0

        self._mu_hist[key].append(float(signed_mu))
        self._sigma_hist[key].append(float(sigma))
        self._latency_hist[key].append(float(latency))
        self._n_obs[key] += 1

        if context_key is not None:
            ctx = self._context_mu[key]

            if context_key not in ctx:
                ctx[context_key] = deque(maxlen=self.window)

            ctx[context_key].append(float(signed_mu))

    def update_from_proxy_output(
        self,
        ego_id: int,
        neighbor_ids: List[int],
        proxy_out: Dict[str, np.ndarray],
        context_keys: Optional[List] = None,
    ):
        """
        Tiện ích: nhận thẳng output của proxy.score_batch_full().

        Args:
            neighbor_ids: list[int] length B — thứ tự khớp với batch của proxy
            proxy_out:    dict từ score_batch_full, cần các khoá mu/sigma/latency
            context_keys: list length B hoặc None
        """
        mu = np.asarray(proxy_out["mu"]).reshape(-1)            # [B]
        sigma = np.asarray(proxy_out["sigma"]).reshape(-1)      # [B]
        latency = np.asarray(
            proxy_out.get("latency", np.zeros_like(mu))
        ).reshape(-1)                                            # [B]

        for b, j in enumerate(neighbor_ids):
            self.update(
                ego_id=ego_id,
                neighbor_id=int(j),
                signed_mu=float(mu[b]),
                sigma=float(sigma[b]),
                latency=float(latency[b]),
                context_key=(
                    None if context_keys is None else context_keys[b]
                ),
            )

    # =====================================================================
    # Truy xuất signature
    # =====================================================================

    def get_signature(self, ego_id: int, neighbor_id: int) -> np.ndarray:
        """
        Returns:
            np.ndarray float32 shape [6] = SIGNATURE_DIM

            [0] signed_mu     trung bình có dấu
            [1] abs_mu        |trung bình|
            [2] sigma         bất định trung bình
            [3] temporal_std  std của mu theo thời gian
            [4] context_std   std của mu-trung-bình-theo-ngữ-cảnh giữa các ngữ cảnh
            [5] latency       trọng tâm horizon trung bình
        """
        key = (int(ego_id), int(neighbor_id))

        if key not in self._mu_hist or len(self._mu_hist[key]) == 0:
            return np.zeros(SIGNATURE_DIM, dtype=np.float32)

        mus = np.asarray(self._mu_hist[key], dtype=np.float64)          # [T]
        sigmas = np.asarray(self._sigma_hist[key], dtype=np.float64)    # [T]
        lats = np.asarray(self._latency_hist[key], dtype=np.float64)    # [T]

        signed_mu = float(np.mean(mus))

        # ---------------------------------------------------------------
        # abs_mu = mean(|mu|), KHÔNG PHẢI |mean(mu)|.
        #
        # Lỗi này bị bắt trong unit test. Nếu để |mean| thì chiều này DƯ THỪA
        # hoàn toàn với signed_mu (chỉ là trị tuyệt đối của nó) -> lãng phí
        # một chiều của signature.
        #
        # Với mean(|·|) ta có hai kênh ĐỘC LẬP và tỷ số giữa chúng mang
        # thông tin thật:
        #     |signed_mu| ~ abs_mu   -> ảnh hưởng NHẤT QUÁN một chiều
        #                               (blocker luôn cản, relay luôn giúp)
        #     |signed_mu| << abs_mu  -> ảnh hưởng MẠNH nhưng ĐẢO CHIỀU
        #                               (lúc giúp lúc hại — loại nguy hiểm,
        #                                đúng đối tượng của ngăn "Dị biệt")
        # Đây chính là trường hợp mà một con số vô hướng không thể diễn tả,
        # và là lý do signature phải nhiều chiều.
        # ---------------------------------------------------------------
        abs_mu = float(np.mean(np.abs(mus)))

        sigma = float(np.mean(sigmas))
        temporal_std = float(np.std(mus)) if mus.size > 1 else 0.0

        # ---- context_std ------------------------------------------------
        # Lấy TRUNG BÌNH TRONG TỪNG ngữ cảnh trước, rồi tính STD GIỮA các
        # ngữ cảnh. Làm vậy để tách "ảnh hưởng khác nhau theo tình huống"
        # ra khỏi "ảnh hưởng nhiễu ngẫu nhiên" (cái sau đã nằm ở temporal_std).
        ctx_map = self._context_mu.get(key, {})
        ctx_means = [
            float(np.mean(np.asarray(v, dtype=np.float64)))
            for v in ctx_map.values()
            if len(v) > 0
        ]

        context_std = (
            float(np.std(np.asarray(ctx_means, dtype=np.float64)))
            if len(ctx_means) > 1
            else 0.0
        )

        latency = float(np.mean(lats))

        return np.array(
            [signed_mu, abs_mu, sigma, temporal_std, context_std, latency],
            dtype=np.float32,
        )

    def get_signature_matrix(
        self,
        ego_id: int,
        neighbor_ids: List[int],
        normalise: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Returns:
            np.ndarray float32 shape [len(neighbor_ids), SIGNATURE_DIM]

        normalise=True: z-score theo từng CỘT (chiều). Cần cho k-means vì
        latency có thang [0, H-1] còn signed_mu có thang ~[-1, 1].
        """
        if normalise is None:
            normalise = self.normalise

        rows = [
            self.get_signature(ego_id, int(j)) for j in neighbor_ids
        ]

        if len(rows) == 0:
            return np.zeros((0, SIGNATURE_DIM), dtype=np.float32)

        mat = np.stack(rows, axis=0).astype(np.float32)  # [N, 6]

        if not normalise or mat.shape[0] < 2:
            return mat

        mean = np.mean(mat, axis=0, keepdims=True)   # [1, 6]
        std = np.std(mat, axis=0, keepdims=True)     # [1, 6]

        return ((mat - mean) / (std + self.eps)).astype(np.float32)

    def get_n_observations(self, ego_id: int, neighbor_id: int) -> int:
        return int(self._n_obs.get((int(ego_id), int(neighbor_id)), 0))

    # =====================================================================
    # GÁN VAI TRÒ — luật ngữ nghĩa
    # =====================================================================

    def get_role(self, ego_id: int, neighbor_id: int) -> int:
        """
        Gán vai trò CỨNG theo luật. Dùng cho logging/diagnostics và để
        khởi tạo slot. Việc gán MỀM (có gradient) nằm ở soft_role_assignment().

        Thứ tự ưu tiên rất quan trọng — Dị biệt xét TRƯỚC:
            nếu ta chưa hiểu rõ thằng này (sigma cao), việc gán nó vào
            Thiện hay Ác đều là võ đoán. Tách riêng ra an toàn hơn.

        Returns:
            int trong {0,1,2,3}
        """
        sig = self.get_signature(ego_id, neighbor_id)

        signed_mu = float(sig[0])
        sigma = float(sig[2])

        if sigma > self.sigma_hi:
            return ROLE_ANOMALOUS

        if abs(signed_mu) < self.tau_role:
            return ROLE_NEUTRAL

        return ROLE_BENEFICIAL if signed_mu > 0.0 else ROLE_HARMFUL

    def get_role_distribution(
        self, ego_id: int, neighbor_ids: List[int]
    ) -> Dict[str, int]:
        """Đếm số neighbour mỗi vai trò — số liệu đưa vào bảng trong paper."""
        counts = {name: 0 for name in ROLE_NAMES}

        for j in neighbor_ids:
            counts[ROLE_NAMES[self.get_role(ego_id, int(j))]] += 1

        return counts

    # =====================================================================
    # Hiệu chỉnh ngưỡng tự động
    # =====================================================================

    def auto_calibrate(
        self,
        ego_ids: Optional[List[int]] = None,
        tau_percentile: float = 50.0,
        sigma_percentile: float = 80.0,
    ):
        """
        Đặt tau_role và sigma_hi theo PHÂN VỊ của dữ liệu thực tế, thay vì
        cắm hằng số.

        Vì sao cần: thang đo của mu phụ thuộc hoàn toàn vào thang reward của
        môi trường. Cắm tau_role = 0.05 có thể khiến TẤT CẢ neighbour rơi vào
        Trung tính (nếu reward nhỏ) hoặc KHÔNG AI vào Trung tính (nếu reward lớn).
        Cả hai đều làm slot ngữ nghĩa vô dụng.

        Gọi hàm này một lần sau Stage 0 warm-up.
        """
        all_abs_mu = []
        all_sigma = []

        for key in self._mu_hist:
            if ego_ids is not None and key[0] not in ego_ids:
                continue

            sig = self.get_signature(key[0], key[1])
            all_abs_mu.append(float(sig[1]))
            all_sigma.append(float(sig[2]))

        if len(all_abs_mu) >= 4:
            self.tau_role = float(
                np.percentile(np.asarray(all_abs_mu), tau_percentile)
            )

        if len(all_sigma) >= 4:
            self.sigma_hi = float(
                np.percentile(np.asarray(all_sigma), sigma_percentile)
            )

        return {
            "tau_role": float(self.tau_role),
            "sigma_hi": float(self.sigma_hi),
            "n_pairs_used": int(len(all_abs_mu)),
        }

    # =====================================================================
    # Nhãn ngoài để đánh giá
    # =====================================================================

    def role_recovery_score(
        self,
        ego_id: int,
        neighbor_ids: List[int],
        ground_truth_roles: Dict[int, int],
    ) -> Dict[str, float]:
        """
        So vai trò khám phá được với vai trò THẬT của môi trường.

        Đây là thí nghiệm then chốt cho paper: kể cả khi reward không tăng,
        nếu phương pháp TỰ KHÁM PHÁ RA vai trò khớp ground truth thì vẫn là
        một kết quả interpretability đăng được.

        Args:
            ground_truth_roles: {neighbor_id: role_int}

        Returns:
            dict với accuracy tổng và accuracy từng vai trò
        """
        correct = 0
        total = 0
        per_role_correct = {r: 0 for r in range(N_SEMANTIC_ROLES)}
        per_role_total = {r: 0 for r in range(N_SEMANTIC_ROLES)}

        for j in neighbor_ids:
            j = int(j)

            if j not in ground_truth_roles:
                continue

            gt = int(ground_truth_roles[j])
            pred = self.get_role(ego_id, j)

            total += 1
            per_role_total[gt] += 1

            if pred == gt:
                correct += 1
                per_role_correct[gt] += 1

        out = {
            "accuracy": float(correct) / float(max(1, total)),
            "n_evaluated": int(total),
        }

        for r in range(N_SEMANTIC_ROLES):
            out[f"acc_{ROLE_NAMES[r]}"] = (
                float(per_role_correct[r]) / float(per_role_total[r])
                if per_role_total[r] > 0
                else float("nan")
            )

        return out

    def get_cluster_centroids(
        self,
        ego_ids: List[int],
        neighbor_ids_per_ego: Dict[int, List[int]],
    ) -> np.ndarray:
        """
        Trung bình signature theo từng vai trò, gộp trên nhiều ego.

        Returns:
            np.ndarray shape [N_SEMANTIC_ROLES, SIGNATURE_DIM]

        HÌNH CHO PAPER: vẽ ma trận này thành heatmap. Nếu bốn hàng khác nhau
        rõ rệt -> chứng minh các slot THẬT SỰ chuyên môn hoá, không collapse.
        Nếu bốn hàng giống nhau -> slot vẫn đang sụp, phải sửa tiếp.
        """
        buckets = {r: [] for r in range(N_SEMANTIC_ROLES)}

        for ego in ego_ids:
            for j in neighbor_ids_per_ego.get(int(ego), []):
                r = self.get_role(int(ego), int(j))
                buckets[r].append(self.get_signature(int(ego), int(j)))

        out = np.zeros((N_SEMANTIC_ROLES, SIGNATURE_DIM), dtype=np.float32)

        for r in range(N_SEMANTIC_ROLES):
            if len(buckets[r]) > 0:
                out[r] = np.mean(
                    np.stack(buckets[r], axis=0), axis=0
                ).astype(np.float32)

        return out


# =========================================================================
# GÁN VAI TRÒ MỀM (có gradient) — dùng trong peripheral_memory_v2
# =========================================================================

def soft_role_assignment(
    signed_mu: np.ndarray,
    sigma: np.ndarray,
    tau_role: float = 0.05,
    sigma_hi: float = 0.5,
    sharpness: float = 3.0,
) -> np.ndarray:
    """
    Phiên bản MỀM của get_role() — trả về phân phối trên 4 vai trò thay vì
    một nhãn cứng.

    Vì sao cần mềm: gán cứng bằng if/else thì gradient không chảy qua được,
    và ở gần biên (mu = tau_role +- epsilon) nhãn nhảy đột ngột gây bất ổn.
    Sigmoid với độ dốc `sharpness` cho chuyển tiếp trơn.

    Cấu trúc cổng:
        g_anom  = sigmoid(k * (sigma - sigma_hi))        "chưa hiểu được"
        g_sure  = 1 - g_anom                             "đã hiểu"
        g_pos   = sigmoid(k * (mu - tau))                "đủ dương"
        g_neg   = sigmoid(k * (-mu - tau))               "đủ âm"
        g_neu   = 1 - g_pos - g_neg  (clamp >= 0)        "quanh 0"

    Args:
        signed_mu: np [N]
        sigma:     np [N]

    Returns:
        np.ndarray float32 shape [N, 4], mỗi hàng tổng bằng 1.
        Cột: [beneficial, harmful, neutral, anomalous]
    """
    mu = np.asarray(signed_mu, dtype=np.float64).reshape(-1)   # [N]
    sg = np.asarray(sigma, dtype=np.float64).reshape(-1)       # [N]

    if mu.shape[0] == 0:
        return np.zeros((0, N_SEMANTIC_ROLES), dtype=np.float32)

    # -----------------------------------------------------------------
    # CHUẨN HOÁ ĐỘ DỐC THEO NGƯỠNG — lỗi này đã bị bắt trong unit test.
    #
    # Nếu dùng thẳng sigmoid(sharpness * (mu - tau)) thì độ dốc phụ thuộc
    # vào THANG ĐO của mu. Với sharpness=10, tau=0.05:
    #     tại mu = 0 (đáng lẽ phải là Trung tính hoàn toàn)
    #     g_pos = sigmoid(10 * (-0.05)) = sigmoid(-0.5) = 0.378  (!!)
    #     -> g_neu = 1 - 0.378 - 0.378 = 0.244 < g_pos
    #     -> mu = 0 bị gán nhầm vào "Thiện". SAI.
    #
    # Sửa: chia sharpness cho ngưỡng, biến nó thành đại lượng KHÔNG THỨ
    # NGUYÊN, nghĩa là "chuyển tiếp diễn ra trong bao nhiêu lần độ rộng tau".
    #     k_mu = sharpness / tau  ->  tại mu=0: sigmoid(-sharpness)
    # Với sharpness=3: sigmoid(-3)=0.047 -> g_neu = 0.906. ĐÚNG.
    # -----------------------------------------------------------------
    k_mu = float(sharpness) / max(float(tau_role), 1e-8)
    k_sg = float(sharpness) / max(float(sigma_hi), 1e-8)

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    g_anom = sigmoid(k_sg * (sg - sigma_hi))       # [N]
    g_sure = 1.0 - g_anom                          # [N]

    g_pos = sigmoid(k_mu * (mu - tau_role))        # [N]
    g_neg = sigmoid(k_mu * (-mu - tau_role))       # [N]
    g_neu = np.clip(1.0 - g_pos - g_neg, 0.0, 1.0)  # [N]

    out = np.zeros((mu.shape[0], N_SEMANTIC_ROLES), dtype=np.float64)

    out[:, ROLE_BENEFICIAL] = g_sure * g_pos
    out[:, ROLE_HARMFUL] = g_sure * g_neg
    out[:, ROLE_NEUTRAL] = g_sure * g_neu
    out[:, ROLE_ANOMALOUS] = g_anom

    # Chuẩn hoá về tổng 1 (g_pos + g_neg + g_neu có thể lệch nhẹ khỏi 1).
    row_sum = np.sum(out, axis=1, keepdims=True)   # [N, 1]
    out = out / np.clip(row_sum, 1e-12, None)

    return out.astype(np.float32)


def kmeans_signature_clusters(
    signatures: np.ndarray,
    n_clusters: int = 4,
    n_iter: int = 25,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means thuần numpy trên ma trận signature — không cần sklearn.

    Dùng cho các SLOT TỰ DO (bổ sung cho slot ngữ nghĩa cố định), để bắt
    những cấu trúc mà bốn luật cứng bỏ sót.

    LƯU Ý VỀ THIẾT KẾ: k-means một mình KHÔNG tốt bằng slot ngữ nghĩa, vì
    ý nghĩa mỗi cụm ĐỔI sau mỗi lần re-cluster -> policy phải học lại từ
    đầu -> thêm một nguồn non-stationarity, đúng thứ paper đang cố diệt.
    Nên dùng kiến trúc LAI: slot ngữ nghĩa cố định + vài slot k-means.

    Args:
        signatures: np [N, D] — NÊN đã chuẩn hoá z-score

    Returns:
        labels:    np int [N]
        centroids: np float32 [n_clusters, D]
    """
    X = np.asarray(signatures, dtype=np.float64)  # [N, D]

    if X.ndim != 2 or X.shape[0] == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((n_clusters, SIGNATURE_DIM), dtype=np.float32),
        )

    N, D = X.shape
    k = int(min(n_clusters, N))

    rng = np.random.RandomState(int(seed))

    # k-means++ khởi tạo (ổn định hơn random nhiều)
    centroids = np.zeros((k, D), dtype=np.float64)
    centroids[0] = X[rng.randint(N)]

    for c in range(1, k):
        d2 = np.min(
            ((X[:, None, :] - centroids[None, :c, :]) ** 2).sum(axis=2),
            axis=1,
        )  # [N]

        total = float(d2.sum())

        if total <= 1e-12:
            centroids[c] = X[rng.randint(N)]
        else:
            centroids[c] = X[rng.choice(N, p=d2 / total)]

    labels = np.zeros(N, dtype=np.int64)

    for _ in range(int(n_iter)):
        # dist: [N, k]
        dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1)  # [N]

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = X[mask].mean(axis=0)

    # Pad nếu n_clusters > N
    if k < n_clusters:
        pad = np.zeros((n_clusters - k, D), dtype=np.float64)
        centroids = np.concatenate([centroids, pad], axis=0)

    return labels.astype(np.int64), centroids.astype(np.float32)
