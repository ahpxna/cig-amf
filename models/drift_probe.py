"""
drift_probe.py — Phát hiện structural shift bằng "nhân chứng đóng băng".

=============================================================================
VẤN ĐỀ CỦA BẢN v1: RESIDUAL TỰ NHIỄM BẨN CHÍNH NÓ
=============================================================================
Bản v1 dùng residual của proxy làm cò súng phát hiện structural shift:

    e_t = | R_hat - R_thuc |

Nhưng proxy đó ĐANG ĐƯỢC HỌC LIÊN TỤC. Hai hệ quả, cả hai đều hỏng:

(1) Thế giới đổi -> proxy lặng lẽ học luôn cái mới -> dự đoán lại đúng
    -> residual nhỏ -> KHÔNG BAO GIỜ báo động. Cái cần phát hiện thì
    proxy đã âm thầm nuốt mất.

(2) Tệ hơn: input của proxy gồm Z^{-j} và M^{-j}, mà hai thứ này phụ thuộc
    vào PARTITION HIỆN TẠI. Đổi partition -> đổi input -> residual tăng
    -> trigger bắn -> đổi partition tiếp. Cò súng đang phản ứng với CHÍNH
    SỰ CỰA QUẬY CỦA MÌNH, không phải với thế giới.

=============================================================================
GIẢI PHÁP: MỘT NHÂN CHỨNG BỊ ĐÓNG BĂNG VÀ BỊT MẮT
=============================================================================
Ví dụ đời thường: bạn muốn biết quán ăn quen có đổi đầu bếp không. Nếu hỏi
người ngày nào cũng ăn ở đó, họ đã quen dần với vị mới nên nói "vẫn thế".
Muốn biết chính xác, phải hỏi người đã ăn quán này 5 năm trước rồi đi xa,
nay quay lại ăn một miếng — vị giác họ vẫn neo ở quá khứ nên phát hiện ngay.

Probe này chính là người đó, với hai đặc điểm:

  ĐÓNG BĂNG (frozen): huấn luyện xong thì khoá trọng số, không học nữa.
      -> ký ức neo ở thời điểm chụp ảnh, không trôi theo thế giới mới.

  BỊT MẮT (context-free): chỉ nhìn (o_i, a_i, a_j). KHÔNG nhìn Z, M, B.
      -> partition đổi kiểu gì cũng không ảnh hưởng tới nó
      -> triệt tiêu hoàn toàn đường nhiễm bẩn (2) ở trên.

Residual của nó vọt lên <=> QUY LUẬT MÔI TRƯỜNG đã đổi, chứ không phải
sổ sách nội bộ của ta đổi.

=============================================================================
LƯU Ý QUAN TRỌNG: PHẢI CHỤP LẠI ẢNH SAU KHI ĐÃ THÍCH NGHI
=============================================================================
Sau một structural shift thật, probe cũ sẽ sai VĨNH VIỄN (thế giới mới,
ký ức cũ) -> nếu không chụp lại, nó báo động mãi không thôi. Vì thế:
    trigger bắn -> hệ thích nghi -> đợi `recalibrate_after` -> chụp ảnh mới
Hàm `maybe_recalibrate()` lo việc này.
=============================================================================
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralDriftProbe(nn.Module):
    """
    Mạng nhỏ dự đoán return của ego CHỈ từ (o_i, a_i, a_j).

    Cố ý làm yếu và bịt mắt: nó không được phép nhìn bất cứ thứ gì phụ thuộc
    vào partition. Nó không cần dự đoán giỏi — nó chỉ cần dự đoán SAI KHÁC ĐI
    khi quy luật môi trường đổi.

    Args:
        obs_dim, action_dim: như proxy chính.
        hidden: nhỏ thôi (64). Mạng lớn dễ overfit và làm residual nhiễu.
        n_horizons: khớp với proxy chính.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 64,
        n_horizons: int = 3,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_horizons = int(n_horizons)

        in_dim = self.obs_dim + 2 * self.action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), self.n_horizons),
        )

    def forward(
        self,
        obs_i: torch.Tensor,            # [B, obs_dim]
        action_i_onehot: torch.Tensor,  # [B, action_dim]
        action_j_onehot: torch.Tensor,  # [B, action_dim]
    ) -> torch.Tensor:
        """Returns: [B, n_horizons]"""
        x = torch.cat([obs_i, action_i_onehot, action_j_onehot], dim=-1)
        return self.net(x)


class DriftDetector:
    """
    Quản lý vòng đời của probe: huấn luyện -> đóng băng -> đo -> chụp lại.

    Cách dùng trong runner:

        det = DriftDetector(obs_dim, action_dim, n_horizons, device)

        # cuối mỗi episode, sau khi đã có buffer:
        det.step(episode, proxy.buffer)
        fired = det.residual_z_score() > threshold   # hoặc dùng scheduler

    Args:
        warmup_batches: số batch huấn luyện trước khi đóng băng lần đầu.
        recalibrate_after: số episode sau một lần bắn thì chụp ảnh lại.
        window: cửa sổ để chuẩn hoá residual thành z-score.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_horizons: int = 3,
        hidden: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        warmup_batches: int = 200,
        batch_size: int = 256,
        recalibrate_after: int = 15,
        window: int = 20,
        seed: int = 0,
    ):
        torch.manual_seed(int(seed))

        self.device = device
        self.action_dim = int(action_dim)
        self.n_horizons = int(n_horizons)
        self.batch_size = int(batch_size)
        self.warmup_batches = int(warmup_batches)
        self.recalibrate_after = int(recalibrate_after)
        self.window = int(window)

        self.live = StructuralDriftProbe(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=hidden,
            n_horizons=n_horizons,
        ).to(device)

        self.optim = torch.optim.Adam(self.live.parameters(), lr=float(lr))

        # Bản đóng băng. None cho tới khi warm-up xong.
        self.frozen: Optional[StructuralDriftProbe] = None

        self.rng = np.random.RandomState(int(seed))

        self.n_batches_trained = 0
        self.residual_history: List[float] = []
        self.last_snapshot_episode = None
        self.pending_recalibration_at = None
        self.n_snapshots = 0

    # ------------------------------------------------------------------

    def _one_hot(self, a: np.ndarray) -> torch.Tensor:
        t = torch.tensor(
            np.asarray(a, dtype=np.int64), dtype=torch.long, device=self.device
        ).clamp(0, self.action_dim - 1)

        return F.one_hot(t, num_classes=self.action_dim).to(dtype=torch.float32)

    def _batch(self, buffer, n: int):
        """Lấy batch từ replay của proxy chính. Trả tensors hoặc None."""
        if buffer is None or len(buffer) == 0:
            return None

        buf = list(buffer)
        n = int(min(n, len(buf)))
        idx = self.rng.choice(len(buf), size=n, replace=False)
        batch = [buf[i] for i in idx]

        obs = torch.tensor(
            np.stack([b["obs_i"] for b in batch], axis=0),
            dtype=torch.float32, device=self.device,
        )                                                    # [B, obs_dim]
        a_i = self._one_hot([b["action_i"] for b in batch])  # [B, A]
        a_j = self._one_hot(
            [b["observed_action_j"] for b in batch]
        )                                                    # [B, A]
        tgt = torch.tensor(
            np.stack([b["target_returns_multi"] for b in batch], axis=0),
            dtype=torch.float32, device=self.device,
        )                                                    # [B, H]

        return obs, a_i, a_j, tgt

    # ------------------------------------------------------------------

    def train_batches(self, buffer, n_batches: int = 1) -> float:
        """Huấn luyện bản LIVE (bản đóng băng không bao giờ được đụng tới)."""
        losses = []

        for _ in range(int(n_batches)):
            got = self._batch(buffer, self.batch_size)

            if got is None:
                break

            obs, a_i, a_j, tgt = got

            self.live.train()
            pred = self.live(obs, a_i, a_j)          # [B, H]
            loss = F.mse_loss(pred, tgt)

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.live.parameters(), 1.0)
            self.optim.step()

            losses.append(float(loss.detach().cpu().item()))
            self.n_batches_trained += 1

        return float(np.mean(losses)) if losses else 0.0

    def snapshot(self, episode: Optional[int] = None):
        """
        Chụp ảnh: sao chép bản live thành bản đóng băng.

        Từ giây phút này, bản đóng băng là "nhân chứng" với ký ức cố định.
        """
        import copy

        self.frozen = copy.deepcopy(self.live).to(self.device)

        for p in self.frozen.parameters():
            p.requires_grad_(False)

        self.frozen.eval()

        self.last_snapshot_episode = episode
        self.pending_recalibration_at = None
        self.n_snapshots += 1

        # Ký ức mới -> thang residual cũ không còn ý nghĩa.
        self.residual_history.clear()

    def measure(self, buffer, n: int = 512) -> Optional[float]:
        """
        Cho bản ĐÓNG BĂNG chấm dữ liệu MỚI NHẤT.

        Lưu ý: lấy từ CUỐI buffer (dữ liệu mới nhất), không lấy ngẫu nhiên,
        vì ta muốn biết "gần đây thế giới có đổi không".

        Returns:
            residual trung bình (MAE trên horizon cuối), hoặc None.
        """
        if self.frozen is None or buffer is None or len(buffer) == 0:
            return None

        buf = list(buffer)[-int(n):]

        obs = torch.tensor(
            np.stack([b["obs_i"] for b in buf], axis=0),
            dtype=torch.float32, device=self.device,
        )
        a_i = self._one_hot([b["action_i"] for b in buf])
        a_j = self._one_hot([b["observed_action_j"] for b in buf])
        tgt = torch.tensor(
            np.stack([b["target_returns_multi"] for b in buf], axis=0),
            dtype=torch.float32, device=self.device,
        )

        with torch.no_grad():
            pred = self.frozen(obs, a_i, a_j)                  # [B, H]
            res = torch.mean(torch.abs(pred[:, -1] - tgt[:, -1]))

        val = float(res.cpu().item())
        self.residual_history.append(val)

        if len(self.residual_history) > 500:
            del self.residual_history[:-500]

        return val

    def residual_z_score(self) -> float:
        """
        Residual mới nhất, quy về số độ lệch chuẩn so với cửa sổ gần đây.

        Dùng z-score thay vì giá trị thô vì thang residual phụ thuộc thang
        reward của môi trường; z-score là đại lượng không thứ nguyên nên
        ngưỡng đặt một lần dùng được ở mọi môi trường.
        """
        h = self.residual_history

        if len(h) < 5:
            return 0.0

        recent = np.asarray(h[-self.window:], dtype=np.float64)
        base = recent[:-1]

        if base.size < 3:
            return 0.0

        mu, sd = float(np.mean(base)), float(np.std(base))

        if sd < 1e-9:
            return 0.0

        return float((h[-1] - mu) / sd)

    # ------------------------------------------------------------------

    def step(self, episode: int, buffer, n_train_batches: int = 5) -> Dict:
        """
        Gọi một lần mỗi episode. Lo toàn bộ vòng đời.

        Returns:
            dict trạng thái để log.
        """
        # Trước khi chụp ảnh lần đầu: huấn luyện bình thường.
        if self.frozen is None:
            self.train_batches(buffer, n_train_batches)

            if self.n_batches_trained >= self.warmup_batches:
                self.snapshot(episode)

            return {
                "phase": "warmup",
                "batches": int(self.n_batches_trained),
                "residual": None,
                "z": 0.0,
            }

        # Sau khi đã có bản đóng băng: bản live VẪN học tiếp, để dành cho
        # lần chụp ảnh sau. Bản đóng băng tuyệt đối không đụng tới.
        self.train_batches(buffer, n_train_batches)

        res = self.measure(buffer)
        z = self.residual_z_score()

        # Đã tới hạn chụp lại chưa?
        if (
            self.pending_recalibration_at is not None
            and episode >= self.pending_recalibration_at
        ):
            self.snapshot(episode)

            return {
                "phase": "recalibrated",
                "batches": int(self.n_batches_trained),
                "residual": res,
                "z": 0.0,
            }

        return {
            "phase": "monitoring",
            "batches": int(self.n_batches_trained),
            "residual": res,
            "z": float(z),
        }

    def notify_trigger(self, episode: int):
        """
        Runner gọi khi trigger đã bắn và hệ đã bắt đầu thích nghi.

        Hẹn lịch chụp ảnh lại. Không có bước này, probe sẽ báo động vĩnh
        viễn sau một shift thật (thế giới mới, ký ức cũ).
        """
        self.pending_recalibration_at = int(episode) + self.recalibrate_after

    def get_diagnostics(self) -> Dict:
        return {
            "n_snapshots": int(self.n_snapshots),
            "n_batches_trained": int(self.n_batches_trained),
            "last_snapshot_episode": self.last_snapshot_episode,
            "pending_recalibration_at": self.pending_recalibration_at,
            "latest_residual": (
                float(self.residual_history[-1])
                if self.residual_history else None
            ),
            "latest_z": float(self.residual_z_score()),
            "frozen_ready": bool(self.frozen is not None),
        }


class MatrixDriftDetector:
    """
    Cò súng thứ hai, độc lập với probe: theo dõi MA TRẬN ẢNH HƯỞNG.

    CƠ SỞ LÝ THUYẾT (mượn Pieroth ICML 2024, Theorem 5.11)
    ------------------------------------------------------
    Họ chứng minh các đại lượng đo ảnh hưởng LIÊN TỤC theo tham số policy.
    Dịch sang ngôn ngữ của ta: behavioural drift (policy trôi từ từ) chỉ có
    thể làm ma trận ảnh hưởng đổi TỪ TỪ. Vậy nếu ma trận NHẢY VỌT, đó không
    thể là behavioural drift — nó BẮT BUỘC phải là structural shift.

    Đây là lập luận toán học cho việc tách hai tầng, không phải trực giác.

    Ưu điểm so với residual probe: không phải đợi return H bước, nên phát
    hiện sớm hơn. Nhược điểm: ma trận tính từ proxy, mà proxy học từ return
    H bước, nên độ trễ vẫn còn nhưng gián tiếp.

    Nên dùng CẢ HAI: probe bắt "quy luật môi trường đổi", ma trận bắt
    "cấu trúc ảnh hưởng đổi". Hai thứ này không phải lúc nào cũng trùng.
    """

    def __init__(self, window: int = 20, eps: float = 1e-8):
        self.window = int(window)
        self.eps = float(eps)

        self.prev: Optional[np.ndarray] = None
        self.history: List[float] = []

    def update(self, W: np.ndarray) -> float:
        """
        W: [n_agents, n_agents] ma trận ảnh hưởng (có dấu).

        Returns:
            độ thay đổi tương đối so với lần trước.
        """
        W = np.asarray(W, dtype=np.float64)

        if self.prev is None:
            self.prev = W.copy()
            return 0.0

        num = float(np.linalg.norm(W - self.prev, ord="fro"))
        den = float(np.linalg.norm(self.prev, ord="fro")) + self.eps

        self.prev = W.copy()

        val = num / den
        self.history.append(val)

        if len(self.history) > 500:
            del self.history[:-500]

        return val

    def z_score(self) -> float:
        if len(self.history) < 5:
            return 0.0

        recent = np.asarray(self.history[-self.window:], dtype=np.float64)
        base = recent[:-1]

        if base.size < 3:
            return 0.0

        mu, sd = float(np.mean(base)), float(np.std(base))

        if sd < 1e-9:
            return 0.0

        return float((self.history[-1] - mu) / sd)

    def get_diagnostics(self) -> Dict:
        return {
            "latest_change": (
                float(self.history[-1]) if self.history else None
            ),
            "latest_z": float(self.z_score()),
            "n_observations": int(len(self.history)),
        }
