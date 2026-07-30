"""
reciprocity.py — Đo chiều NGƯỢC: tôi có ảnh hưởng nó không?

=============================================================================
Ý TƯỞNG
=============================================================================
Toàn bộ phần còn lại của CIG-AMF đo MỘT chiều:

    w_ij  :  j -> i     "nó ảnh hưởng tôi bao nhiêu"

File này đo chiều còn lại:

    g_ij  :  i -> j     "tôi ảnh hưởng nó bao nhiêu"

Cách đo, nói bằng lời: huấn luyện HAI mô hình cùng dự đoán hành động kế
tiếp của j.
    Mô hình A -- CHỈ nhìn j          (dùng shadow state s_ij)
    Mô hình B -- nhìn CẢ j VÀ tôi    (dùng pair latent z_ij)

Nếu B đoán giỏi hơn A, nghĩa là biết về tôi giúp đoán được j
  -> j ĐANG PHẢN ỨNG VỚI TÔI.
Nếu hai mô hình ngang nhau -> j hoàn toàn phớt lờ tôi.

Hiệu số chính là information gain (lượng thông tin thu được):

    g_ij = CE(mô hình A) - CE(mô hình B)     [nats]

CE là cross-entropy, tức "độ ngỡ ngàng trung bình khi thấy hành động thật".
g_ij > 0 nghĩa là biết thêm về tôi làm giảm sự ngỡ ngàng.

=============================================================================
HẠ TẦNG ĐÃ CÓ SẴN — ĐÂY LÀ LÝ DO NÓ RẺ
=============================================================================
    z_ij  cập nhật từ (z_prev, o_i, a_i, o_j, a_j, xi)  <- CÓ ego
    s_ij  cập nhật từ (s_prev,        o_j, a_j, xi)     <- KHÔNG ego

Hai biểu diễn này đã tồn tại trong CIG-AMF vì lý do khác (shadow dùng để
warm-start khi promote). Chúng vô tình tạo thành đúng cặp đối chứng cần
thiết. Ta chỉ phải thêm MỘT đầu tuyến tính trên shadow. Khoảng 40 dòng.

=============================================================================
VẤN ĐỀ CONFOUNDING VÀ CÁCH GIẢI
=============================================================================
Bản quan sát thuần bị nhiễu nặng: nếu i và j đứng cạnh nhau, o_i và o_j
nhìn thấy CÙNG một thứ. Mô hình B đoán giỏi hơn không phải vì j phản ứng
với i, mà vì o_i vô tình chứa thông tin về hoàn cảnh chung.

Đây đúng là cái bẫy correlation-vs-causation mà cả bài đang chống.

CÁCH GIẢI, và nó gần như MIỄN PHÍ: chỉ tính trên những bước mà hành động
của CHÍNH EGO i bị eps-forcing ép ngẫu nhiên (F_i = 1). Ở những bước đó,
a_i độc lập với mọi thứ khác theo đúng nghĩa cơ học. Nếu a_i vẫn giúp đoán
được a_j^{t+1}, thì đó BẮT BUỘC là nhân quả.

Ta đã ép mọi agent sẵn rồi, nên chỉ cần lọc theo cờ F_i.

=============================================================================
Ô 2x2 — ĐÂY MỚI LÀ ĐÓNG GÓP KHÁI NIỆM
=============================================================================
Ghép hai chiều lại được bốn loại quan hệ:

                     |  i->j THẤP (nó kệ tôi)  |  i->j CAO (nó để ý tôi)
    -----------------|-------------------------|--------------------------
    j->i THẤP        |  vô can                 |  KẺ BÁM ĐUÔI
    (tôi kệ nó)      |                         |  (nhìn tôi mà không hại tôi)
    -----------------|-------------------------|--------------------------
    j->i CAO         |  VẬT CẢN VÔ TRI         |  CẶP GHÉP CHIẾN LƯỢC  (!)
    (nó ảnh hưởng tôi)|  (chắn đường, không    |  (hai bên cùng đọc bài nhau)
                     |   biết tôi tồn tại)     |

Ô dưới-phải là nơi non-stationarity thật sự sinh ra: hai agent liên tục
thích nghi với nhau, tạo thành vòng lặp đuổi nhau mà sách giáo khoa MARL
gọi là "cyclic dynamics". Nếu đo được ô này, ta chỉ đích danh được các cặp
CHỊU TRÁCH NHIỆM cho sự phi dừng -- điều chưa ai làm.

Ô dưới-trái quan trọng theo cách khác: vật cản vô tri thì chỉ cần TRÁNH,
không cần mô hình hoá hành vi -- vì nó không phản ứng với ta. Ngược lại,
cặp ghép chiến lược thì bắt buộc phải mô hình hoá kỹ.

=============================================================================
LIÊN HỆ VỚI JAQUES (2019)
=============================================================================
Social influence của Jaques đo đúng chiều i->j nhưng trên HÀNH ĐỘNG và
bằng KL quan sát, rồi dùng làm intrinsic reward. Ở đây:
    - chiều i->j: gần giống họ, nhưng đo bằng can thiệp thật (F_i=1)
    - chiều j->i: trên RETURN của ego, thứ họ không đo
    - mục đích: phân loại quan hệ, không phải thưởng
Nói cách khác, ta thu hồi đại lượng của Jaques như MỘT TRỤC trong hồ sơ
hai chiều. Đây là cách định vị rất sạch so với tổ tiên gần nhất.

=============================================================================
ĐÁNH GIÁ THẲNG THẮN VỀ MỨC ĐỘ HỨA HẸN
=============================================================================
ĐIỂM MẠNH
  + rẻ thật (hạ tầng có sẵn, ~40 dòng)
  + bản nhân quả gần như miễn phí nhờ eps-forcing đã có
  + tạo ra một hình 2x2 mới, gắn thẳng vào luận đề trung tâm của bài
  + định vị sạch so với Jaques

RỦI RO
  - trong môi trường resource-flow hợp tác, có thể ĐA SỐ cặp rơi vào ô
    "vật cản vô tri", ô ghép chiến lược rỗng -> hình đẹp nhưng vô nghĩa
  - mẫu mỏng: cần F_i = 1, mà eps chỉ 3% -> mỗi cặp rất ít mẫu nhân quả
  - chưa rõ policy DÙNG con số này để làm gì

KẾT LUẬN
  Triển khai như CÔNG CỤ CHẨN ĐOÁN trước, KHÔNG cắm vào vòng lặp quyết
  định. Vẽ hình 2x2. Chỉ khi hình đó cho thấy cấu trúc thật thì mới đưa
  vào core selection ở bài sau. Cách này lấy được phần lợi (hình mới, định
  vị mới) mà không đánh cược cả bài vào một giả thuyết chưa kiểm chứng.

  Vì thế mặc định `use_in_signature=False`.
=============================================================================
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # cho phép import file này khi chưa cài torch
    _HAS_TORCH = False
    nn = object


# Nhãn của bốn ô
QUAD_IRRELEVANT = 0     # vô can
QUAD_FOLLOWER = 1       # kẻ bám đuôi
QUAD_OBSTACLE = 2       # vật cản vô tri
QUAD_STRATEGIC = 3      # cặp ghép chiến lược

QUAD_NAMES = ("irrelevant", "follower", "obstacle", "strategic")
QUAD_NAMES_VI = ("vô can", "bám đuôi", "vật cản vô tri", "ghép chiến lược")


if _HAS_TORCH:

    class EgoFreeActionHead(nn.Module):
        """
        Mô hình A: dự đoán a_j^{t+1} CHỈ từ shadow state (không có ego).

        Cố ý giữ cùng dung lượng với đầu dự đoán trên z_ij, nếu không thì
        hiệu số cross-entropy sẽ phản ánh chênh lệch dung lượng mạng chứ
        không phản ánh lượng thông tin về ego. Đây là chi tiết dễ bỏ sót
        nhưng làm hỏng toàn bộ phép đo.
        """

        def __init__(self, shadow_dim: int, action_dim: int, hidden: int = 64):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(int(shadow_dim), int(hidden)),
                nn.ReLU(),
                nn.Linear(int(hidden), int(action_dim)),
            )

        def forward(self, s: "torch.Tensor") -> "torch.Tensor":
            """s: [B, shadow_dim] -> logits [B, action_dim]"""
            return self.net(s)


class ReciprocityTracker:
    """
    Theo dõi information gain hai chiều cho từng cặp có hướng.

    Cách dùng trong runner (sau khi đã có z_ij và s_ij của bước t, và
    đã biết hành động thật a_j ở bước t+1):

        rec.record(
            ego_id=i, neighbor_id=j,
            logits_with_ego=head_z(z_ij),        # [action_dim]
            logits_without_ego=head_shadow(s_ij),# [action_dim]
            true_next_action=a_j_next,
            ego_was_forced=bool(forced_mask_prev[i]),
        )

    Rồi lấy kết quả:
        g = rec.get_gain(i, j, causal_only=True)
        quad = rec.get_quadrant(i, j, w_ij=belief.debiased_mu(j))

    Args:
        window: số quan sát gần nhất giữ cho mỗi cặp.
        min_causal_samples: dưới ngưỡng này thì bản nhân quả coi như chưa
            đủ tin cậy và trả về NaN thay vì một con số bịa.
    """

    def __init__(
        self,
        n_agents: int,
        window: int = 200,
        min_causal_samples: int = 20,
        eps: float = 1e-8,
    ):
        self.n_agents = int(n_agents)
        self.window = int(window)
        self.min_causal_samples = int(min_causal_samples)
        self.eps = float(eps)

        # ce[(i,j)] = deque các (ce_without, ce_with, was_forced)
        self._ce: Dict[Tuple[int, int], deque] = {}

    # ------------------------------------------------------------------

    @staticmethod
    def _cross_entropy(logits: np.ndarray, target: int) -> float:
        """
        CE của một mẫu, tính ổn định số học (log-sum-exp).

        Đây là "độ ngỡ ngàng": mô hình gán xác suất càng thấp cho hành động
        thật thì CE càng cao.
        """
        z = np.asarray(logits, dtype=np.float64).reshape(-1)
        z = z - np.max(z)
        logZ = float(np.log(np.sum(np.exp(z)) + 1e-300))
        t = int(np.clip(target, 0, z.shape[0] - 1))

        return float(logZ - z[t])

    def record(
        self,
        ego_id: int,
        neighbor_id: int,
        logits_with_ego,
        logits_without_ego,
        true_next_action: int,
        ego_was_forced: bool = False,
    ):
        """Ghi nhận một quan sát cho cặp (i, j)."""
        key = (int(ego_id), int(neighbor_id))

        if key not in self._ce:
            self._ce[key] = deque(maxlen=self.window)

        if _HAS_TORCH and isinstance(logits_with_ego, torch.Tensor):
            logits_with_ego = logits_with_ego.detach().cpu().numpy()

        if _HAS_TORCH and isinstance(logits_without_ego, torch.Tensor):
            logits_without_ego = logits_without_ego.detach().cpu().numpy()

        ce_with = self._cross_entropy(logits_with_ego, true_next_action)
        ce_without = self._cross_entropy(logits_without_ego, true_next_action)

        self._ce[key].append(
            (float(ce_without), float(ce_with), bool(ego_was_forced))
        )

    # ------------------------------------------------------------------

    def get_gain(
        self,
        ego_id: int,
        neighbor_id: int,
        causal_only: bool = True,
    ) -> float:
        """
        Information gain g_ij = CE_without - CE_with, đơn vị nats.

        Args:
            causal_only:
                True  -> CHỈ dùng các bước mà hành động của ego bị ép ngẫu
                         nhiên. Ở đó a_i độc lập với mọi thứ, nên mọi khả
                         năng dự đoán đều là nhân quả. Trả NaN nếu chưa đủ
                         mẫu (thà không biết còn hơn biết sai).
                False -> dùng tất cả. Rẻ và luôn có số, nhưng DÍNH
                         CONFOUNDING: o_i và o_j nhìn cùng một thế giới.

        Returns:
            float (có thể là NaN nếu causal_only và thiếu mẫu)
        """
        key = (int(ego_id), int(neighbor_id))
        rows = self._ce.get(key)

        if not rows:
            return float("nan") if causal_only else 0.0

        if causal_only:
            sel = [(a, b) for a, b, f in rows if f]

            if len(sel) < self.min_causal_samples:
                return float("nan")
        else:
            sel = [(a, b) for a, b, _ in rows]

        arr = np.asarray(sel, dtype=np.float64)      # [n, 2]

        return float(np.mean(arr[:, 0] - arr[:, 1]))

    def get_n_causal_samples(self, ego_id: int, neighbor_id: int) -> int:
        rows = self._ce.get((int(ego_id), int(neighbor_id)))

        if not rows:
            return 0

        return int(sum(1 for _, _, f in rows if f))

    # ------------------------------------------------------------------

    def get_quadrant(
        self,
        ego_id: int,
        neighbor_id: int,
        w_ij: float,
        tau_w: float = 0.05,
        tau_g: float = 0.02,
        causal_only: bool = True,
    ) -> int:
        """
        Xếp cặp (i, j) vào một trong bốn ô.

        Args:
            w_ij: ảnh hưởng j->i (lấy từ belief.debiased_mu(j)).
            tau_w, tau_g: ngưỡng cho hai trục. Nên đặt bằng phân vị của
                phân phối quan sát được, không cắm cứng (xem calibrate).

        Returns:
            int trong {0,1,2,3}, hoặc -1 nếu chưa đủ dữ liệu nhân quả.
        """
        g = self.get_gain(ego_id, neighbor_id, causal_only=causal_only)

        if not np.isfinite(g):
            return -1

        strong_in = abs(float(w_ij)) > float(tau_w)   # nó ảnh hưởng tôi
        strong_out = float(g) > float(tau_g)          # tôi ảnh hưởng nó

        if strong_in and strong_out:
            return QUAD_STRATEGIC

        if strong_in:
            return QUAD_OBSTACLE

        if strong_out:
            return QUAD_FOLLOWER

        return QUAD_IRRELEVANT

    def calibrate(
        self,
        percentile: float = 70.0,
        causal_only: bool = True,
    ) -> Dict[str, float]:
        """
        Đặt ngưỡng tau_g theo phân vị của chính dữ liệu quan sát được.

        Cần thiết vì thang của information gain phụ thuộc entropy của không
        gian hành động: với 5 hành động, gain tối đa là log(5) ~ 1.61 nats.
        Cắm cứng 0.02 có thể đúng ở môi trường này và sai hoàn toàn ở
        môi trường khác.
        """
        vals = []

        for (i, j) in self._ce:
            g = self.get_gain(i, j, causal_only=causal_only)

            if np.isfinite(g):
                vals.append(g)

        if len(vals) < 4:
            return {"tau_g": 0.02, "n_pairs": len(vals), "calibrated": False}

        return {
            "tau_g": float(np.percentile(np.asarray(vals), percentile)),
            "n_pairs": int(len(vals)),
            "mean_gain": float(np.mean(vals)),
            "max_gain": float(np.max(vals)),
            "calibrated": True,
        }

    # ------------------------------------------------------------------

    def quadrant_report(
        self,
        belief_modules: Dict,
        tau_w: float = 0.05,
        tau_g: float = 0.02,
        causal_only: bool = True,
    ) -> Dict:
        """
        Quét toàn bộ cặp, đếm số lượng mỗi ô, và trả về dữ liệu để vẽ.

        HÌNH CHO PAPER: scatter với trục x = w_ij (j->i) và trục y = g_ij
        (i->j), chia bốn góc phần tư bằng hai đường tau. Nếu các điểm tụ
        thành cụm rõ ở nhiều ô -> có cấu trúc thật, đáng đưa vào cơ chế.
        Nếu tất cả dồn vào một ô -> ý tưởng không cho thêm thông tin gì,
        và nên nói thẳng như vậy trong Discussion.

        Returns:
            dict gồm counts, points (để vẽ), và tỷ lệ mẫu đủ tin cậy.
        """
        counts = {name: 0 for name in QUAD_NAMES}
        counts["insufficient_data"] = 0

        points = []

        for ego_id, mod in belief_modules.items():
            for j in getattr(mod, "neighbor_ids", []):
                if int(ego_id) == int(j):
                    continue

                w = float(mod.debiased_mu(int(j)))
                g = self.get_gain(int(ego_id), int(j), causal_only=causal_only)

                q = self.get_quadrant(
                    int(ego_id), int(j), w_ij=w,
                    tau_w=tau_w, tau_g=tau_g, causal_only=causal_only,
                )

                if q < 0:
                    counts["insufficient_data"] += 1
                    continue

                counts[QUAD_NAMES[q]] += 1
                points.append({
                    "ego": int(ego_id), "neighbor": int(j),
                    "w_ij": w, "g_ij": float(g), "quadrant": int(q),
                })

        total = max(1, sum(counts.values()))

        return {
            "counts": counts,
            "points": points,
            "n_pairs": int(total),
            "coverage": float(1.0 - counts["insufficient_data"] / total),
            "tau_w": float(tau_w),
            "tau_g": float(tau_g),
        }

    def get_diagnostics(self) -> Dict:
        n_pairs = len(self._ce)
        n_causal = [
            self.get_n_causal_samples(i, j) for (i, j) in self._ce
        ]

        return {
            "n_pairs_tracked": int(n_pairs),
            "mean_causal_samples": (
                float(np.mean(n_causal)) if n_causal else 0.0
            ),
            "min_causal_samples_seen": int(min(n_causal)) if n_causal else 0,
            "pairs_with_enough_causal": int(
                sum(1 for c in n_causal if c >= self.min_causal_samples)
            ),
        }
