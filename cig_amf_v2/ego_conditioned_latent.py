"""
ego_conditioned_latent.py — Vá cho z_ij (pair-specific relational latent).

=============================================================================
LỖI CỦA BẢN v1
=============================================================================
Paper tuyên bố z_ij nắm bắt "how neighbour j's behaviour relates SPECIFICALLY
to ego-agent i's outcome, rather than representing j through a single global
latent" — đó là ĐÓNG GÓP SỐ 2 trong bốn đóng góp của bài.

Nhưng hàm mất mát (Eq. 16, hiện thực ở core_behavior.py::train_bc) là:

    L_z = -log p(a_j^{t+1} | z_ij^t)
    # code: F.cross_entropy(self.bc_head(z_next), target_a_j)

VIỆC DỰ ĐOÁN a_j KHÔNG CẦN BẤT KỲ THÔNG TIN NÀO VỀ i.

Input x_full có chứa o_i và a_i, nhưng gradient sẽ đẩy encoder tới chỗ
PHỚT LỜ chúng, vì chúng không giúp giảm loss. Kết quả: z_ij hội tụ về một
opponent model TOÀN CỤC giống hệt nhau cho mọi ego.

=> Tính "pair-specific" bị VÔ HIỆU HOÁ NGAY TỪ HÀM MẤT MÁT. Đóng góp số 2
   hiện không được bảo vệ bởi bất kỳ cơ chế nào.

Test để tự kiểm chứng (nên chạy trước khi sửa, đưa vào paper làm bằng chứng):
    đo cosine similarity giữa z_ij và z_i'j với i != i', cùng một j.
    Nếu ~1.0 -> latent KHÔNG pair-specific, chỉ là global opponent model.
    Hàm pair_specificity_score() ở cuối file làm đúng việc này.

=============================================================================
BA HẠNG THỨC BỔ SUNG
=============================================================================

[E1] INFLUENCE HEAD — dự đoán w_ij từ z_ij.
     Đây là cách trực tiếp nhất: bắt z_ij phải mang thông tin về "j ảnh
     hưởng i thế nào". Khác với a_j, đại lượng w_ij PHỤ THUỘC CẢ HAI PHÍA
     nên encoder buộc phải dùng o_i, a_i.
     Nguồn: tinh thần difference rewards / COMA, nhưng áp lên cặp có hướng.

[E2] CONTRASTIVE EGO — ép z_ij khác z_i'j.
     Cùng một neighbour j, nhìn từ hai ego khác nhau phải cho latent khác
     nhau (trừ khi ảnh hưởng thật sự giống nhau). InfoNCE với positive là
     cùng cặp ở bước thời gian khác, negative là cùng j nhưng khác ego.
     Nguồn: ACORM dùng contrastive GIỮA CÁC AGENT; ta dùng GIỮA CÁC EGO
     CHO CÙNG MỘT AGENT -> khác chiều tương phản, là mượn-có-cải-tiến.

[E3] Giữ nguyên L_z gốc (dự đoán a_j) làm hạng thức phụ.
     Nó vẫn hữu ích để bám behavioural drift — chỉ là không đủ một mình.

    L_total = L_bc + w_inf * L_influence + w_con * L_contrastive
=============================================================================
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EgoConditionedHeads(nn.Module):
    """
    Các đầu dự đoán bổ sung gắn lên z_ij có sẵn.

    Thiết kế để CẮM THÊM vào PairRelationalModule hiện có mà KHÔNG phải
    viết lại nó. Cách dùng trong runner:

        heads = EgoConditionedHeads(latent_dim=pair_rel.hidden_dim, ...)
        opt   = torch.optim.Adam(heads.parameters(), lr=1e-3)

        # trong vòng train, sau khi có z_batch:
        loss = heads.compute_loss(z_batch, ego_ids, neighbor_ids, w_targets)
        opt.zero_grad(); loss.backward(); opt.step()

    Args:
        latent_dim:
            số chiều của z_ij (= pair_rel_module.hidden_dim).
        n_horizons:
            nếu > 1, influence head dự đoán w theo từng horizon -> ép z_ij
            mang cả thông tin ĐỘ TRỄ, không chỉ độ lớn.
        proj_dim:
            số chiều không gian chiếu cho contrastive.
        temperature:
            nhiệt độ InfoNCE. Nhỏ -> phạt mạnh các cặp gần nhau.
    """

    def __init__(
        self,
        latent_dim: int,
        n_horizons: int = 3,
        hidden: int = 64,
        proj_dim: int = 32,
        temperature: float = 0.2,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.n_horizons = int(n_horizons)
        self.temperature = float(temperature)

        # [E1] z_ij -> w_ij (có dấu, theo từng horizon)
        self.influence_head = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), self.n_horizons),
        )

        # [E2] z_ij -> không gian chiếu cho contrastive
        self.proj_head = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(proj_dim)),
        )

    # =====================================================================
    # [E1] Influence prediction
    # =====================================================================

    def influence_loss(
        self,
        z: torch.Tensor,            # [B, latent_dim]
        w_target: torch.Tensor,     # [B, n_horizons] hoặc [B]
    ) -> torch.Tensor:
        """
        Ép z_ij dự đoán được ảnh hưởng nhân quả CÓ DẤU của j lên i.

        Vì sao hạng thức này làm cho "pair-specific" thành THẬT:
        w_ij phụ thuộc vào CẢ i lẫn j. Muốn dự đoán được nó, encoder BẮT BUỘC
        phải dùng o_i và a_i trong input — đúng thứ mà loss cũ (dự đoán a_j)
        cho phép nó phớt lờ.

        Returns: scalar
        """
        if z.shape[0] == 0:
            return torch.zeros((), dtype=torch.float32, device=z.device)

        pred = self.influence_head(z)  # [B, n_horizons]

        if w_target.dim() == 1:
            target = w_target.unsqueeze(1).expand(-1, self.n_horizons)
        else:
            target = w_target

        # Huber ổn định hơn MSE khi w có outlier (proxy đôi khi nhảy).
        return F.smooth_l1_loss(pred, target)

    # =====================================================================
    # [E2] Contrastive ego
    # =====================================================================

    def contrastive_loss(
        self,
        z: torch.Tensor,               # [B, latent_dim]
        ego_ids: torch.Tensor,         # [B] long
        neighbor_ids: torch.Tensor,    # [B] long
    ) -> torch.Tensor:
        """
        InfoNCE ép z_ij TÁCH KHỎI z_i'j (cùng neighbour j, khác ego i).

        Cách xây cặp:
            anchor  : mẫu b
            positive: mẫu khác CÙNG (ego, neighbour)  -> nên GẦN
            negative: mẫu CÙNG neighbour, KHÁC ego    -> nên XA

        Chỉ dùng negative "cùng j khác i" chứ không phải mọi mẫu khác. Lý do:
        ta không muốn ép z_ij khác z_ik (hai neighbour khác nhau) — điều đó
        đã tự nhiên rồi. Cái ta cần ép là CÙNG MỘT NEIGHBOUR nhìn từ hai ego
        khác nhau phải cho biểu diễn khác nhau. Đó chính là định nghĩa của
        "ego-centric role": j có thể là blocker với i nhưng relay với i'.

        Returns: scalar
        """
        B = z.shape[0]

        if B < 3:
            return torch.zeros((), dtype=torch.float32, device=z.device)

        p = F.normalize(self.proj_head(z), p=2, dim=1, eps=1e-8)  # [B, proj_dim]
        sim = (p @ p.t()) / self.temperature                       # [B, B]

        same_ego = ego_ids.view(-1, 1) == ego_ids.view(1, -1)      # [B, B]
        same_nb = neighbor_ids.view(-1, 1) == neighbor_ids.view(1, -1)  # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=z.device)      # [B, B]

        pos_mask = same_ego & same_nb & (~eye)     # cùng cặp, khác thời điểm
        neg_mask = (~same_ego) & same_nb           # cùng j, khác ego

        # Chỉ tính cho các anchor có ĐỦ cả positive lẫn negative.
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)  # [B]

        if not bool(valid.any()):
            return torch.zeros((), dtype=torch.float32, device=z.device)

        neg_inf = torch.finfo(sim.dtype).min

        pos_sim = sim.masked_fill(~pos_mask, neg_inf)  # [B, B]
        cand_mask = pos_mask | neg_mask                 # [B, B]
        all_sim = sim.masked_fill(~cand_mask, neg_inf)  # [B, B]

        # InfoNCE: -log( sum exp(pos) / sum exp(pos + neg) )
        log_num = torch.logsumexp(pos_sim[valid], dim=1)   # [n_valid]
        log_den = torch.logsumexp(all_sim[valid], dim=1)   # [n_valid]

        return torch.mean(log_den - log_num)

    # =====================================================================
    # Tổng hợp
    # =====================================================================

    def compute_loss(
        self,
        z: torch.Tensor,                       # [B, latent_dim]
        ego_ids: torch.Tensor,                 # [B]
        neighbor_ids: torch.Tensor,            # [B]
        w_target: Optional[torch.Tensor] = None,   # [B, H] hoặc [B]
        w_influence: float = 1.0,
        w_contrastive: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict:
            total       : scalar — cộng vào L_bc bên ngoài
            influence   : scalar
            contrastive : scalar
        """
        device = z.device
        zero = torch.zeros((), dtype=torch.float32, device=device)

        l_inf = (
            self.influence_loss(z, w_target) if w_target is not None else zero
        )
        l_con = self.contrastive_loss(z, ego_ids, neighbor_ids)

        total = w_influence * l_inf + w_contrastive * l_con

        return {"total": total, "influence": l_inf, "contrastive": l_con}


# =========================================================================
# Chẩn đoán: latent có THẬT SỰ pair-specific không?
# =========================================================================

def pair_specificity_score(
    pair_rel_module,
    n_agents: int,
    sample_neighbors: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    ĐO XEM z_ij CÓ THẬT SỰ PAIR-SPECIFIC HAY KHÔNG.

    Chạy hàm này TRƯỚC khi vá và SAU khi vá. Cặp số liệu này là bằng chứng
    trực tiếp rằng đóng góp số 2 của paper đã được hiện thực hoá — hiện tại
    paper KHÔNG có bất kỳ số liệu nào chứng minh điều này.

    Cách đo:
        Với mỗi neighbour j, lấy tập {z_ij : mọi ego i != j}.
        Tính cosine similarity trung bình GIỮA CÁC EGO.

        ~1.0 -> mọi ego thấy j giống hệt nhau
                => z chỉ là GLOBAL opponent model, KHÔNG pair-specific
        thấp -> các ego thấy j khác nhau
                => THẬT SỰ ego-centric (điều paper tuyên bố)

    So sánh với baseline: similarity giữa các NEIGHBOUR KHÁC NHAU cùng một
    ego. Nếu cross_ego_similarity ~ cross_neighbor_similarity thì latent
    không phân biệt được gì cả.

    Returns:
        dict với cross_ego_similarity, cross_neighbor_similarity, ratio
    """
    ids = (
        list(range(int(n_agents)))
        if sample_neighbors is None
        else [int(x) for x in sample_neighbors]
    )

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))

        if na < 1e-8 or nb < 1e-8:
            return 0.0

        return float(np.dot(a, b) / (na * nb))

    # ---- cùng neighbour j, khác ego --------------------------------------
    cross_ego = []

    for j in ids:
        vecs = []

        for i in range(int(n_agents)):
            if i == j:
                continue

            v = pair_rel_module.get_pair_latent(i, j)

            if v is not None and np.linalg.norm(v) > 1e-8:
                vecs.append(np.asarray(v, dtype=np.float64).reshape(-1))

        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                cross_ego.append(_cos(vecs[a], vecs[b]))

    # ---- cùng ego i, khác neighbour --------------------------------------
    cross_nb = []

    for i in ids:
        vecs = []

        for j in range(int(n_agents)):
            if i == j:
                continue

            v = pair_rel_module.get_pair_latent(i, j)

            if v is not None and np.linalg.norm(v) > 1e-8:
                vecs.append(np.asarray(v, dtype=np.float64).reshape(-1))

        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                cross_nb.append(_cos(vecs[a], vecs[b]))

    ce = float(np.mean(cross_ego)) if cross_ego else 0.0
    cn = float(np.mean(cross_nb)) if cross_nb else 0.0

    return {
        "cross_ego_similarity": ce,
        "cross_neighbor_similarity": cn,
        # < 1 nghĩa là cùng-j-khác-ego đã tách ra được tốt hơn
        # so với khác-j-cùng-ego. Đó là dấu hiệu ego-centric thật.
        "specificity_ratio": float(ce / (cn + 1e-8)),
        "n_cross_ego_pairs": int(len(cross_ego)),
        "n_cross_neighbor_pairs": int(len(cross_nb)),
    }
