"""
peripheral_memory_v2.py — Peripheral multi-memory, bản vá slot collapse.

=============================================================================
CHẨN ĐOÁN NULL RESULT CỦA BẢN v1
=============================================================================
Bảng kết quả trong paper:
    NoMultiMemory (BỎ multi-memory)  Core F1 = 0.251 / 0.259, throughput 241.8
    Final CIG-AMF (CÓ multi-memory)  Core F1 = 0.244 / 0.262, throughput  74.7

Bỏ module chính đi thì NGANG BẰNG mà NHANH GẤP 3.2 LẦN. Tức module đang
chiếm ~70% chi phí tính toán mà không mang lại gì.

NGUYÊN NHÂN — SLOT COLLAPSE. Nhìn code v1:

    slot_logits = self.slot_router(enc_in)      # không ai giao việc
    slot_probs  = F.softmax(slot_logits, dim=-1)

Không có BẤT KỲ tín hiệu huấn luyện nào nói "ngăn 1 chứa loại này, ngăn 2
chứa loại kia". Mạng chỉ được bảo "chia kiểu gì cũng được, miễn reward tốt".
Mà reward thì xa xôi, nhiễu, gradient truyền về tận lớp gán-ngăn cực kỳ yếu.

Khi không ai giao việc, mạng tìm đường dễ nhất. Hai kịch bản sụp:
  (a) Sụp ĐỒNG PHỤC: softmax gán mọi item vào 4 ngăn với trọng số ~đều
      (mỗi ngăn ~25%) -> cả 4 ngăn chứa cùng hỗn hợp -> mỗi ngăn ~ trung bình
      toàn bộ -> concat 4 ngăn = 4 bản photocopy của single mean.
      Bốn nồi lẩu nhưng CÙNG MỘT VỊ.
  (b) Sụp ĐỘC QUYỀN: một ngăn hút hết, ba ngăn còn lại rỗng -> cũng ~ mean.

v1 còn có `uniform_mix = 0.25` trộn thêm uniform memory vào — điều này
LÀM TỆ THÊM kịch bản (a): nó chủ động kéo mọi slot về gần trung bình chung.

=============================================================================
BA LỚP THUỐC CHỮA (dùng đồng thời)
=============================================================================

[T1] GIAO VIỆC CHO NGĂN — slot ngữ nghĩa.
     Thay vì để softmax tự bơi, mỗi ngăn được ĐỊNH NGHĨA TRƯỚC bằng vai trò
     chức năng suy từ chữ ký ảnh hưởng nhân quả:
        ngăn 0 "Thiện"     : mu > 0, mạnh, chắc chắn
        ngăn 1 "Ác"        : mu < 0, mạnh, chắc chắn
        ngăn 2 "Trung tính": |mu| ~ 0
        ngăn 3 "Dị biệt"   : sigma cao — chưa hiểu được, "bọn làm mình sợ"
     Gán MỀM bằng sigmoid nên gradient vẫn chảy.

     Ưu điểm so với k-means: ý nghĩa ngăn CỐ ĐỊNH qua thời gian. K-means
     phải re-cluster định kỳ và ý nghĩa cụm đổi mỗi lần -> policy học lại
     từ đầu -> thêm một nguồn non-stationarity, đúng thứ paper đang diệt.

     Ngăn "Dị biệt" đáng chú ý: nó biến sigma từ một tham số điều khiển
     thành MỘT CHIỀU NGỮ NGHĨA. Hầu hết phương pháp coi uncertainty là thứ
     cần GIẢM; ở đây nó là THUỘC TÍNH ĐỂ PHÂN LOẠI.

[T2] CHỐNG SỤP ĐỘC QUYỀN — load-balancing loss (Switch Transformer).
        L_lb = alpha * K * sum_q  f_q * P_q
     f_q = tỷ lệ item được định tuyến vào ngăn q
     P_q = xác suất định tuyến trung bình router gán cho ngăn q
     Cả hai bằng 1/K khi cân bằng; tích nhỏ nhất khi đều.
     Gradient tỉ lệ mức quá tải -> vòng phản hồi tự sửa.
     Fedus et al. quét alpha từ 1e-1 đến 1e-5, khuyến nghị 1e-2.

[T3] CHỐNG SỤP ĐỒNG PHỤC — orthogonality loss.
        L_orth = mean_{q != r} cosine_similarity(m_q, m_r)^2
     Phạt khi vector của các ngăn quá giống nhau. Đây chính là "chống 4 nồi
     cùng một vị". Load-balancing KHÔNG chữa được kịch bản (a) vì phân bổ
     có thể rất đều mà nội dung vẫn giống hệt nhau — nên cần cả hai.

     (Bổ sung tuỳ chọn: MI regularizer kiểu ROMA, xem slot_specialisation_loss)
=============================================================================
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.influence_signature import (
    N_SEMANTIC_ROLES,
    ROLE_ANOMALOUS,
    ROLE_BENEFICIAL,
    ROLE_HARMFUL,
    ROLE_NEUTRAL,
)


class PeripheralMultiMemory(nn.Module):
    """
    Peripheral encoder với slot ngữ nghĩa + slot tự do.

    Kiến trúc LAI:
        - N_SEMANTIC_ROLES = 4 slot CỐ ĐỊNH theo vai trò (Thiện/Ác/Trung tính/Dị biệt)
        - n_free_slots slot TỰ DO học bằng router (bắt cái mà 4 luật bỏ sót)
    Tổng số slot = 4 + n_free_slots.

    item format (GIỮ NGUYÊN 9 chiều như v1 để runner không phải sửa):
        0: action_j
        1: mu_bar        <- GIỜ CÓ DẤU (v1 luôn >= 0)
        2: sigma_bar
        3: p_core
        4: in_prev_core
        5: rel_row
        6: rel_col
        7: zone_diff
        8: distance_norm

    Input:
        periph_items: np/tensor [N_p, 9]
    Output:
        forward()      -> [out_dim]
        forward_full() -> dict gồm memory + các loss phụ trợ

    Args:
        n_free_slots:
            số slot tự do ngoài 4 slot ngữ nghĩa. 0 = chỉ dùng slot ngữ nghĩa.
        lb_coeff:
            hệ số load-balancing loss. Fedus et al. khuyến nghị 1e-2.
        orth_coeff:
            hệ số orthogonality loss.
        role_sharpness:
            độ dốc sigmoid khi gán mềm. Lớn -> gần gán cứng.
        use_uniform_mix:
            v1 trộn uniform memory (mặc định 0.25) — điều này KÉO MỌI SLOT
            VỀ GẦN TRUNG BÌNH CHUNG, tức chủ động gây sụp đồng phục.
            v2 mặc định TẮT. Bật lại chỉ để chạy ablation "trước khi vá".
    """

    def __init__(
        self,
        action_dim: int,
        memory_dim: int = 32,
        out_dim: int = 64,
        item_hidden: int = 48,
        item_dim: int = 9,
        n_free_slots: int = 2,
        # ---- ngưỡng vai trò (nên lấy từ tracker.auto_calibrate()) ----
        tau_role: float = 0.05,
        sigma_hi: float = 0.5,
        role_sharpness: float = 3.0,
        # ---- hệ số regulariser ----
        lb_coeff: float = 0.5,
        orth_coeff: float = 1e-2,
        # ---- tương thích ngược ----
        num_slots: Optional[int] = None,
        use_uniform_mix: bool = True,
        uniform_mix: float = 0.25,
        mu_floor: float = 0.02,
        beta_floor: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.action_dim = int(action_dim)
        self.memory_dim = int(memory_dim)
        self.out_dim = int(out_dim)
        self.item_hidden = int(item_hidden)
        self.item_dim = int(item_dim)

        self.n_semantic_slots = int(N_SEMANTIC_ROLES)
        self.n_free_slots = int(max(0, n_free_slots))

        # `num_slots` của v1 nếu được truyền vào thì hiểu là TỔNG số slot.
        if num_slots is not None:
            total = int(num_slots)
            self.n_free_slots = int(max(0, total - self.n_semantic_slots))

        self.num_slots = self.n_semantic_slots + self.n_free_slots

        self.tau_role = float(tau_role)
        self.sigma_hi = float(sigma_hi)
        self.role_sharpness = float(role_sharpness)

        self.lb_coeff = float(lb_coeff)
        self.orth_coeff = float(orth_coeff)

        self.use_uniform_mix = bool(use_uniform_mix)
        self.uniform_mix = float(uniform_mix)
        self.mu_floor = float(mu_floor)
        self.beta_floor = float(beta_floor)
        self.eps = float(eps)

        # action one-hot + phần còn lại của item
        self.non_action_dim = self.item_dim - 1
        self.encoder_in_dim = self.action_dim + self.non_action_dim

        self.item_encoder = nn.Sequential(
            nn.Linear(self.encoder_in_dim, self.item_hidden),
            nn.ReLU(),
            nn.Linear(self.item_hidden, self.memory_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------
        # Router CHỈ phụ trách slot tự do. Slot ngữ nghĩa gán bằng luật.
        #
        # ROUTER ĐƯỢC ĐIỀU KIỆN HOÁ THEO NHÓM NGỮ NGHĨA — thiết kế này đã
        # được kiểm chứng bằng thực nghiệm trên dữ liệu tổng hợp 4 vai trò
        # (blocker / relay / consumer / inert):
        #     k-means TOÀN CỤC trên signature   : purity 0.767
        #     ngữ nghĩa + k-means TRONG TỪNG NHÓM: purity 0.967
        # Lý do: chuẩn hoá toàn cục bị chi phối bởi các vai trò ảnh hưởng
        # MẠNH (blocker, relay), khiến các vai trò YẾU (consumer vs inert)
        # bị nén sát vào nhau và không tách được. Khi phân nhóm theo dấu
        # trước rồi mới tách trong nhóm, việc tách blocker/consumer trong
        # nhóm "Ác" đạt purity 1.000.
        #
        # Cách hiện thực hoá mà vẫn giữ tính khả vi: nối sem_probs vào input
        # của router, để router biết item thuộc nhóm nào và chuyên môn hoá
        # theo nhóm — thay vì phải phân nhóm cứng rồi chạy k-means rời rạc.
        # ---------------------------------------------------------------
        if self.n_free_slots > 0:
            self.router_in_dim = self.encoder_in_dim + self.n_semantic_slots

            self.slot_router = nn.Sequential(
                nn.Linear(self.router_in_dim, self.item_hidden),
                nn.ReLU(),
                nn.Linear(self.item_hidden, self.n_free_slots),
            )
        else:
            self.router_in_dim = self.encoder_in_dim
            self.slot_router = None

        self.out_proj = nn.Sequential(
            nn.Linear(self.num_slots * self.memory_dim, self.out_dim),
            nn.ReLU(),
        )

        # Chẩn đoán slot usage — số PHẢI báo cáo để chứng minh hết collapse.
        self.register_buffer(
            "slot_usage_ema",
            torch.full((self.num_slots,), 1.0 / float(self.num_slots)),
        )
        self.usage_ema_alpha = 0.05

    # =====================================================================
    # Helper
    # =====================================================================

    def _device(self):
        return next(self.parameters()).device

    def _one_hot_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [N] long -> [N, action_dim]"""
        a = actions.long().clamp(min=0, max=self.action_dim - 1)
        return F.one_hot(a, num_classes=self.action_dim).to(dtype=torch.float32)

    def _normalise_inputs(self, periph_items) -> torch.Tensor:
        device = self._device()

        if periph_items is None:
            return torch.zeros(0, self.item_dim, dtype=torch.float32, device=device)

        if isinstance(periph_items, np.ndarray):
            x = torch.from_numpy(periph_items).to(device=device, dtype=torch.float32)
        elif isinstance(periph_items, torch.Tensor):
            x = periph_items.to(device=device, dtype=torch.float32)
        else:
            x = torch.tensor(
                np.asarray(periph_items, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.numel() == 0:
            return torch.zeros(0, self.item_dim, dtype=torch.float32, device=device)

        if x.shape[-1] != self.item_dim:
            raise ValueError(
                f"PeripheralMultiMemory expected item_dim={self.item_dim}, "
                f"got {x.shape[-1]}"
            )

        return x

    def _prepare_encoder_input(self, items: torch.Tensor) -> torch.Tensor:
        """items: [N, 9] -> [N, action_dim + 8]"""
        action_col = items[:, 0].long().clamp(min=0, max=self.action_dim - 1)
        action_oh = self._one_hot_actions(action_col)  # [N, action_dim]
        rest = items[:, 1:].to(dtype=torch.float32)    # [N, 8]

        return torch.cat([action_oh, rest], dim=-1)    # [N, action_dim+8]

    # =====================================================================
    # [T1] GÁN SLOT NGỮ NGHĨA (mềm, có gradient)
    # =====================================================================

    def _semantic_slot_probs(self, items: torch.Tensor) -> torch.Tensor:
        """
        Gán mềm mỗi peripheral item vào 4 vai trò từ chữ ký ảnh hưởng.

        items: [N, 9]  (cột 1 = mu_bar CÓ DẤU, cột 2 = sigma_bar)

        Returns:
            [N, 4] — mỗi hàng tổng ~1

        Cấu trúc cổng (khớp với soft_role_assignment trong influence_signature.py):
            g_anom = sigmoid(k*(sigma - sigma_hi))       chưa hiểu được
            g_sure = 1 - g_anom
            g_pos  = sigmoid(k*(mu - tau))               đủ dương -> Thiện
            g_neg  = sigmoid(k*(-mu - tau))              đủ âm    -> Ác
            g_neu  = clamp(1 - g_pos - g_neg, 0, 1)      quanh 0  -> Trung tính

        Thứ tự ưu tiên: Dị biệt xét TRƯỚC. Nếu chưa hiểu rõ thằng này thì
        gán nó vào Thiện hay Ác đều là võ đoán.
        """
        mu = items[:, 1]                       # [N]  CÓ DẤU
        sigma = torch.clamp(items[:, 2], min=0.0)  # [N]

        # Độ dốc PHẢI chuẩn hoá theo ngưỡng, nếu không thì tại mu=0 hàm
        # sigmoid chưa kịp bão hoà và "Trung tính" bị thua "Thiện"/"Ác".
        # (Lỗi này đã bị bắt trong unit test — xem influence_signature.py.)
        k_mu = self.role_sharpness / max(self.tau_role, 1e-8)
        k_sg = self.role_sharpness / max(self.sigma_hi, 1e-8)

        g_anom = torch.sigmoid(k_sg * (sigma - self.sigma_hi))  # [N]
        g_sure = 1.0 - g_anom                                   # [N]

        g_pos = torch.sigmoid(k_mu * (mu - self.tau_role))      # [N]
        g_neg = torch.sigmoid(k_mu * (-mu - self.tau_role))     # [N]
        g_neu = torch.clamp(1.0 - g_pos - g_neg, min=0.0, max=1.0)  # [N]

        probs = torch.zeros(
            items.shape[0], self.n_semantic_slots,
            dtype=torch.float32, device=items.device,
        )  # [N, 4]

        probs[:, ROLE_BENEFICIAL] = g_sure * g_pos
        probs[:, ROLE_HARMFUL] = g_sure * g_neg
        probs[:, ROLE_NEUTRAL] = g_sure * g_neu
        probs[:, ROLE_ANOMALOUS] = g_anom

        row_sum = torch.clamp(probs.sum(dim=1, keepdim=True), min=self.eps)  # [N,1]

        return probs / row_sum  # [N, 4]

    def _importance_beta(self, items: torch.Tensor) -> torch.Tensor:
        """
        Trọng số tin cậy khi pooling trong mỗi slot.

        v1 dùng |mu| ở tử -> item ảnh hưởng mạnh được ưu tiên. GIỮ NGUYÊN
        tinh thần đó, nhưng lấy trị tuyệt đối Ở ĐÂY (không phải trong
        estimator), vì tới bước này ta đã dùng xong DẤU để chọn slot.

        beta = (beta_floor + p_core) * (|mu| + mu_floor) * 1/(1+sigma)

        Returns: [N]
        """
        mu = items[:, 1]                             # [N] có dấu
        sigma = torch.clamp(items[:, 2], min=0.0)    # [N]
        p_core = torch.clamp(items[:, 3], min=0.0, max=1.0)  # [N]

        confidence = 1.0 / (1.0 + sigma + self.eps)  # [N]

        beta = (
            (self.beta_floor + p_core)
            * (torch.abs(mu) + self.mu_floor)
            * confidence
        )  # [N]

        return torch.clamp(beta, min=self.eps)

    # =====================================================================
    # [T2][T3] Các loss phụ trợ
    # =====================================================================

    def _load_balancing_loss(self, slot_probs: torch.Tensor) -> torch.Tensor:
        """
        [T2] Switch Transformer load-balancing loss.

            L = K * sum_q  f_q * P_q

        slot_probs: [N, K]

        f_q: tỷ lệ item mà ngăn q là ngăn ĐƯỢC CHỌN (argmax) — đây là đại
             lượng rời rạc nên không có gradient, dùng như hệ số.
        P_q: xác suất định tuyến TRUNG BÌNH — đây là chỗ gradient chảy qua.

        Khi cân bằng hoàn hảo: f_q = P_q = 1/K -> L = K * K * (1/K^2) = 1.
        Càng mất cân bằng L càng lớn.

        Returns: scalar
        """
        N, K = slot_probs.shape

        if N == 0:
            return torch.zeros((), dtype=torch.float32, device=slot_probs.device)

        # f_q: one-hot của argmax rồi lấy trung bình. Không cần gradient.
        with torch.no_grad():
            hard = F.one_hot(
                slot_probs.argmax(dim=1), num_classes=K
            ).to(dtype=torch.float32)  # [N, K]
            f = hard.mean(dim=0)       # [K]

        P = slot_probs.mean(dim=0)     # [K]  <- gradient chảy qua đây

        return float(K) * torch.sum(f * P)

    def _orthogonality_loss(self, memories: torch.Tensor) -> torch.Tensor:
        """
        [T3] Phạt khi các slot vector quá giống nhau — chống "4 nồi cùng vị".

        memories: [K, memory_dim]

        L = mean over q != r of  cos(m_q, m_r)^2

        Bình phương cosine chứ không phải cosine thuần: ta muốn phạt cả
        giống hệt (cos=1) lẫn đối nhau hoàn toàn (cos=-1)? KHÔNG — đối nhau
        là ổn (chúng phân biệt được). Nhưng dùng bình phương giúp loss luôn
        không âm và phạt mạnh khi |cos| gần 1. Với slot ngữ nghĩa Thiện/Ác,
        ta KỲ VỌNG chúng đối nhau, nên có tuỳ chọn dùng abs thay vì square
        nếu muốn cho phép đối cực (xem allow_antipodal).

        Returns: scalar
        """
        K = memories.shape[0]

        if K < 2:
            return torch.zeros((), dtype=torch.float32, device=memories.device)

        normed = F.normalize(memories, p=2, dim=1, eps=self.eps)  # [K, D]
        gram = normed @ normed.t()                                 # [K, K]

        # Lấy phần ngoài đường chéo
        mask = ~torch.eye(K, dtype=torch.bool, device=memories.device)  # [K, K]
        off_diag = gram[mask]                                            # [K*(K-1)]

        return torch.mean(off_diag ** 2)

    # =====================================================================
    # Forward
    # =====================================================================

    def forward_full(self, periph_items) -> Dict[str, torch.Tensor]:
        """
        Forward đầy đủ, trả cả memory lẫn các loss phụ trợ.

        Returns dict:
            memory:      [out_dim]           — cắm vào policy
            lb_loss:     scalar              — load balancing
            orth_loss:   scalar              — orthogonality
            aux_loss:    scalar              — lb_coeff*lb + orth_coeff*orth
            slot_probs:  [N, K]              — để chẩn đoán
            slot_usage:  [K]                 — tỷ lệ dùng mỗi ngăn
            memories:    [K, memory_dim]     — vector từng ngăn
        """
        items = self._normalise_inputs(periph_items)  # [N, 9]
        device = self._device()

        zero = torch.zeros((), dtype=torch.float32, device=device)

        # ---- rỗng ------------------------------------------------------
        if items.shape[0] == 0:
            x = torch.zeros(
                1, self.num_slots * self.memory_dim,
                dtype=torch.float32, device=device,
            )
            return {
                "memory": self.out_proj(x).squeeze(0),   # [out_dim]
                "lb_loss": zero,
                "orth_loss": zero,
                "aux_loss": zero,
                "slot_probs": torch.zeros(
                    0, self.num_slots, dtype=torch.float32, device=device
                ),
                "slot_usage": torch.zeros(
                    self.num_slots, dtype=torch.float32, device=device
                ),
                "memories": torch.zeros(
                    self.num_slots, self.memory_dim,
                    dtype=torch.float32, device=device,
                ),
            }

        N = items.shape[0]

        enc_in = self._prepare_encoder_input(items)   # [N, action_dim+8]
        h = self.item_encoder(enc_in)                 # [N, memory_dim]

        # ---- [T1] slot ngữ nghĩa ---------------------------------------
        sem_probs = self._semantic_slot_probs(items)  # [N, 4]

        # ---- slot tự do (ĐIỀU KIỆN HOÁ THEO NHÓM NGỮ NGHĨA) -------------
        if self.n_free_slots > 0:
            # Nối sem_probs vào input: router biết item thuộc nhóm nào
            # -> học được cấu trúc TRONG từng nhóm (blocker vs consumer),
            # thay vì phải tách chúng trong không gian toàn cục nơi chúng
            # bị nén sát nhau. Xem ghi chú ở __init__ về kết quả thực nghiệm.
            router_in = torch.cat(
                [enc_in, sem_probs.detach()], dim=-1
            )  # [N, encoder_in_dim + 4]

            free_logits = self.slot_router(router_in)             # [N, n_free]
            free_probs = F.softmax(free_logits, dim=-1)           # [N, n_free]

            # Trộn: mỗi item chia đôi khối lượng giữa phần ngữ nghĩa và
            # phần tự do. 0.5/0.5 là mặc định trung tính.
            slot_probs = torch.cat(
                [0.5 * sem_probs, 0.5 * free_probs], dim=1
            )  # [N, K]
        else:
            slot_probs = sem_probs  # [N, 4]

        # ---- pooling từng slot ------------------------------------------
        beta = self._importance_beta(items)           # [N]

        # weighted[n, q] = slot_probs[n,q] * beta[n]
        weighted = slot_probs * beta.unsqueeze(1)     # [N, K]

        # memories[q] = sum_n weighted[n,q] * h[n] / sum_n weighted[n,q]
        num = weighted.t() @ h                        # [K, memory_dim]
        den = torch.clamp(
            weighted.sum(dim=0), min=self.eps
        ).unsqueeze(1)                                # [K, 1]

        memories = num / den                          # [K, memory_dim]

        # ---- uniform mix (v1) — mặc định TẮT ----------------------------
        # Bật lại chỉ để chạy ablation. Nó chủ động kéo mọi slot về gần
        # trung bình chung, tức GÂY sụp đồng phục.
        if self.use_uniform_mix:
            num_u = slot_probs.t() @ h                              # [K, D]
            den_u = torch.clamp(
                slot_probs.sum(dim=0), min=self.eps
            ).unsqueeze(1)                                          # [K, 1]
            uniform_mem = num_u / den_u                             # [K, D]

            mix = float(np.clip(self.uniform_mix, 0.0, 1.0))
            memories = (1.0 - mix) * memories + mix * uniform_mem

        # ---- loss phụ trợ ------------------------------------------------
        lb_loss = self._load_balancing_loss(slot_probs)
        orth_loss = self._orthogonality_loss(memories)
        aux_loss = self.lb_coeff * lb_loss + self.orth_coeff * orth_loss

        # ---- chẩn đoán usage --------------------------------------------
        with torch.no_grad():
            usage = slot_probs.mean(dim=0)  # [K]
            self.slot_usage_ema.mul_(1.0 - self.usage_ema_alpha).add_(
                self.usage_ema_alpha * usage
            )

        flat = memories.reshape(1, -1)                # [1, K*memory_dim]
        memory_out = self.out_proj(flat).squeeze(0)   # [out_dim]

        return {
            "memory": memory_out,
            "lb_loss": lb_loss,
            "orth_loss": orth_loss,
            "aux_loss": aux_loss,
            "slot_probs": slot_probs,
            "slot_usage": usage,
            "memories": memories,
        }

    def forward_excluding_all(self, periph_items, item_ids) -> Dict[int, torch.Tensor]:
        """
        [GPU_OPTIMIZATION_CONTRACT.md mục 2.1] M_i^{-j} cho MỌI j trong tập
        peripheral hiện tại của một ego, CÙNG LÚC, bằng thủ thuật sum-trừ-một
        — thay vì gọi forward_full() riêng cho từng exclusion (bản cũ:
        build_inputs + forward đầy đủ N lần mỗi ego, tức chạy lại
        item_encoder/slot_router cho gần hết tập N-1 lần nữa mỗi lần).

        CHỈ ĐÚNG VỚI POOLING KIỂU WEIGHTED-SUM (Eq. 25 — mean pooling có
        trọng số, permutation-invariant kiểu Deep Sets), vì mỗi item đóng
        góp qua h[n]/slot_probs[n]/beta[n] ĐỘC LẬP, không có chuẩn hoá chéo
        item nào trước bước pooling (đã kiểm: item_encoder/semantic gate/
        free-slot router đều là MLP áp per-item, không có BatchNorm/attention
        giữa các item). Nếu sau này đổi pooling sang attention hoặc max
        (paper nhắc Set Transformer là biến thể tương lai), hàm này SAI —
        phải quay lại forward_full() riêng từng exclusion.

        KHÔNG dùng hàm này để train (không tính lb_loss/orth_loss) — chỉ
        phục vụ dựng context M_i^{-j} làm input cho proxy. Huấn luyện
        periph_module vẫn qua forward_full() trên tập ĐẦY ĐỦ.

        Args:
            periph_items: [N, item_dim] — toàn bộ tập peripheral hiện tại.
            item_ids: list[int] độ dài N, id neighbour tương ứng từng hàng.

        Returns:
            {item_id: memory_out [out_dim]} cho mọi id trong item_ids.
        """
        items = self._normalise_inputs(periph_items)
        N = items.shape[0]

        if N == 0:
            return {}

        with torch.no_grad():
            enc_in = self._prepare_encoder_input(items)   # [N, enc_in_dim]
            h = self.item_encoder(enc_in)                 # [N, D]
            sem_probs = self._semantic_slot_probs(items)  # [N, 4]

            if self.n_free_slots > 0:
                router_in = torch.cat([enc_in, sem_probs.detach()], dim=-1)
                free_logits = self.slot_router(router_in)
                free_probs = F.softmax(free_logits, dim=-1)
                slot_probs = torch.cat(
                    [0.5 * sem_probs, 0.5 * free_probs], dim=1
                )  # [N, K]
            else:
                slot_probs = sem_probs  # [N, K]

            beta = self._importance_beta(items)          # [N]
            weighted = slot_probs * beta.unsqueeze(1)     # [N, K]

            num = weighted.t() @ h                         # [K, D]  tổng ĐẦY ĐỦ
            den = weighted.sum(dim=0)                       # [K]

            # Đóng góp riêng từng item vào từng slot -> [N, K, D], vector
            # hoá qua chiều N thay vì vòng lặp Python.
            contrib = weighted.unsqueeze(2) * h.unsqueeze(1)   # [N, K, D]
            num_excl = num.unsqueeze(0) - contrib               # [N, K, D]
            den_excl = torch.clamp(
                den.unsqueeze(0) - weighted, min=self.eps
            ).unsqueeze(2)                                       # [N, K, 1]
            memories = num_excl / den_excl                        # [N, K, D]

            if self.use_uniform_mix:
                num_u = slot_probs.t() @ h                          # [K, D]
                den_u = slot_probs.sum(dim=0)                        # [K]
                contrib_u = slot_probs.unsqueeze(2) * h.unsqueeze(1)  # [N,K,D]
                num_u_excl = num_u.unsqueeze(0) - contrib_u
                den_u_excl = torch.clamp(
                    den_u.unsqueeze(0) - slot_probs, min=self.eps
                ).unsqueeze(2)
                uniform_mem = num_u_excl / den_u_excl                  # [N,K,D]

                mix = float(np.clip(self.uniform_mix, 0.0, 1.0))
                memories = (1.0 - mix) * memories + mix * uniform_mem

            flat = memories.reshape(N, -1)          # [N, K*memory_dim]
            outs = self.out_proj(flat)              # [N, out_dim]

        return {int(item_ids[n]): outs[n] for n in range(N)}

    def forward(self, periph_items) -> torch.Tensor:
        """
        GIỮ NGUYÊN chữ ký v1 — trả đúng [out_dim].
        Runner cũ gọi được ngay; muốn dùng aux_loss thì gọi forward_full().
        """
        return self.forward_full(periph_items)["memory"]

    # =====================================================================
    # build_inputs — giữ nguyên chữ ký v1
    # =====================================================================

    def build_inputs(
        self,
        ego_id,
        peripheral_ids,
        env,
        belief_state,
        prev_core_set=None,
    ) -> np.ndarray:
        """
        Dựng ma trận item cho một ego-agent.

        KHÁC v1 DUY NHẤT MỘT CHỖ: mu_bar giờ CÓ DẤU (v1 luôn >= 0 vì proxy
        đã lấy abs). Đây là thứ khiến slot ngữ nghĩa Thiện/Ác hoạt động được.

        Returns:
            np.ndarray float32 [len(peripheral_ids), 9]
        """
        ego_id = int(ego_id)
        prev_core_set = set() if prev_core_set is None else set(prev_core_set)

        ids = [int(j) for j in list(peripheral_ids) if int(j) != ego_id]

        if len(ids) == 0:
            return np.zeros((0, self.item_dim), dtype=np.float32)

        pi = env.positions[ego_id]
        grid_den = max(1, int(env.grid_size))
        zone_den = max(1, int(env.n_zones) - 1)

        last_actions = getattr(
            env, "last_actions", [0] * int(env.n_agents)
        )

        rows = []

        for j in ids:
            pj = env.positions[j]
            b = belief_state[j]

            action_j = int(np.clip(int(last_actions[j]), 0, self.action_dim - 1))

            rows.append([
                float(action_j),
                float(b["mu_bar"]),      # CÓ DẤU
                float(b["sigma_bar"]),
                float(b["p_core"]),
                float(j in prev_core_set),
                float((pj[0] - pi[0]) / grid_den),
                float((pj[1] - pi[1]) / grid_den),
                float((env.agent_zone[j] - env.agent_zone[ego_id]) / zone_den),
                float(abs(pj[0] - pi[0]) + abs(pj[1] - pi[1])) / grid_den,
            ])

        return np.asarray(rows, dtype=np.float32)

    # =====================================================================
    # Chẩn đoán
    # =====================================================================

    def get_slot_diagnostics(self) -> Dict[str, float]:
        """
        BẰNG CHỨNG CHỐNG COLLAPSE — phải đưa vào paper.

        usage_entropy_ratio:
            entropy của phân bố usage / log(K).
            ~1.0 = dùng đều cả K ngăn (tốt)
            ~0.0 = một ngăn hút hết (sụp độc quyền)
            LƯU Ý: chỉ số này CAO KHÔNG đủ để kết luận không collapse — sụp
            ĐỒNG PHỤC cho entropy = 1.0 hoàn hảo mà vẫn vô dụng. Phải xem
            kèm orthogonality (vẽ heatmap centroid).

        max_usage / min_usage:
            nếu min_usage ~ 0 thì có ngăn chết.
        """
        usage = self.slot_usage_ema.detach().cpu().numpy()  # [K]
        K = int(usage.shape[0])

        p = np.clip(usage / max(float(usage.sum()), 1e-12), 1e-12, 1.0)
        entropy = float(-np.sum(p * np.log(p)))
        max_entropy = float(np.log(K)) if K > 1 else 1.0

        out = {
            "n_slots": K,
            "n_semantic_slots": int(self.n_semantic_slots),
            "n_free_slots": int(self.n_free_slots),
            "usage_entropy_ratio": float(entropy / max(max_entropy, 1e-12)),
            "max_usage": float(np.max(usage)),
            "min_usage": float(np.min(usage)),
            "lb_coeff": float(self.lb_coeff),
            "orth_coeff": float(self.orth_coeff),
        }

        for q in range(min(K, self.n_semantic_slots)):
            name = ("beneficial", "harmful", "neutral", "anomalous")[q]
            out[f"usage_{name}"] = float(usage[q])

        return out

    def set_role_thresholds(self, tau_role: float, sigma_hi: float):
        """
        Cập nhật ngưỡng sau khi gọi tracker.auto_calibrate().

        QUAN TRỌNG: thang của mu phụ thuộc hoàn toàn vào thang reward của
        môi trường. Cắm cứng tau_role có thể khiến TẤT CẢ neighbour rơi vào
        Trung tính (reward nhỏ) hoặc KHÔNG AI vào Trung tính (reward lớn) —
        cả hai đều làm slot ngữ nghĩa vô dụng.
        """
        self.tau_role = float(tau_role)
        self.sigma_hi = float(sigma_hi)


# =========================================================================
# MI regularizer kiểu ROMA (tuỳ chọn, thay cho orthogonality)
# =========================================================================

def slot_specialisation_loss(
    slot_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Regulariser chuyên môn hoá kiểu ROMA, dạng thông tin tương hỗ.

        I(slot; item) = H(E_n[p(slot|n)]) - E_n[H(p(slot|n))]

    Ý nghĩa hai số hạng:
      - H(trung bình): entropy của phân bố usage tổng. CAO là tốt
        -> dùng đều mọi ngăn (chống sụp độc quyền).
      - E[H]: entropy trung bình của từng gán. THẤP là tốt
        -> mỗi item được gán DỨT KHOÁT vào một ngăn, không nhoè
        (chống sụp đồng phục).

    Tối đa hoá I <=> tối thiểu hoá -I. Hàm này trả về -I để dùng làm loss.

    Đây là chỗ mượn từ ROMA (Wang et al. ICML 2020): họ dùng MI để ép
    role <-> trajectory. Ta ép slot <-> influence signature. Khác NGUỒN
    TÍN HIỆU (interventional vs observational), nên là mượn-có-cải-tiến
    chứ không phải bê nguyên xi.

    Args:
        slot_probs: [N, K]

    Returns:
        scalar (= -I, càng nhỏ càng chuyên môn hoá tốt)
    """
    if slot_probs.shape[0] == 0:
        return torch.zeros((), dtype=torch.float32, device=slot_probs.device)

    p = torch.clamp(slot_probs, min=eps)             # [N, K]

    marginal = p.mean(dim=0)                          # [K]
    h_marginal = -torch.sum(marginal * torch.log(marginal + eps))

    h_conditional = -torch.mean(torch.sum(p * torch.log(p), dim=1))

    mutual_info = h_marginal - h_conditional

    return -mutual_info


# Alias tương thích ngược.
PeripheralMultiMemory = PeripheralMultiMemory
