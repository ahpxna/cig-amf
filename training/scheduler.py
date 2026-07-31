"""
scheduler_v2.py — Điều phối hai timescale + phát hiện shift + bơm phồng.

Khác bản v1 ở ba chỗ:
  1. Nhận tín hiệu từ HAI cò súng độc lập (probe đóng băng + ma trận ảnh
     hưởng) thay vì một residual tự nhiễm bẩn.
  2. Khi bắn, không chỉ tăng tốc cập nhật mà còn BƠM PHỒNG BẤT ĐỊNH của
     belief -> core tự co lại -> hệ thận trọng khi chưa hiểu tình hình mới.
  3. Có thời gian trơ (refractory) để không bắn liên hồi trong lúc hệ đang
     thích nghi -- nếu không, mỗi lần core đổi lại kích hoạt một lần bắn nữa.
"""

from typing import Dict, List, Optional

import numpy as np


class TwoTimescaleScheduler:
    """
    Giữ nguyên API v1 (`step_episode`, `should_update_graph`, `in_warmup`,
    `get_status`) nên runner cũ gọi được. Thêm `evaluate_drift`.

    Args:
        k0_warmup: số episode Stage 0.
        alpha_slow_ratio: 0.05 -> cập nhật cấu trúc mỗi ~20 episode.
        accel_factor / accel_duration: mức và thời lượng tăng tốc.
        z_threshold: ngưỡng z-score để coi là có shift. 2.5-3.0 hợp lý
            (z=3 nghĩa là "lệch 3 độ lệch chuẩn so với gần đây").
        require_both: True = phải CẢ HAI cò súng đồng ý mới bắn (ít báo
            động giả, phát hiện chậm hơn). False = một cái là đủ.
        refractory: số episode trơ sau mỗi lần bắn.
        inflation_factor: hệ số nhân sigma khi bắn.
    """

    def __init__(
        self,
        k0_warmup: int = 20,
        alpha_fast: float = 1e-3,
        alpha_slow_ratio: float = 0.05,
        accel_factor: float = 4.0,
        accel_duration: int = 8,
        z_threshold: float = 3.0,
        require_both: bool = False,
        refractory: int = 10,
        inflation_factor: float = 2.5,
        inflation_t_reset: int = 1,
    ):
        self.k0_warmup = int(k0_warmup)
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow_ratio = float(alpha_slow_ratio)
        self.alpha_slow_base = self.alpha_fast * self.alpha_slow_ratio
        self.alpha_slow = 0.0

        self.accel_factor = float(accel_factor)
        self.accel_duration = int(accel_duration)

        self.z_threshold = float(z_threshold)
        self.require_both = bool(require_both)
        self.refractory = int(refractory)

        self.inflation_factor = float(inflation_factor)
        self.inflation_t_reset = int(inflation_t_reset)

        self.episode = 0
        self.stage = 0

        self.accel_remaining = 0
        self.trigger_count = 0
        self.last_trigger_episode = None
        self.trigger_log: List[Dict] = []

        # Lịch sử residual thô, chỉ dùng cho wrapper tương thích ngược
        # `record_structural_residual()` (API v1). Tách riêng khỏi mọi
        # thứ dùng cho `evaluate_drift()` để không lẫn hai cơ chế.
        self._residual_history: List[float] = []
        self._residual_window = 20

    # ------------------------------------------------------------------
    # API v1
    # ------------------------------------------------------------------

    def in_warmup(self) -> bool:
        return self.stage == 0

    def in_learned_stage(self) -> bool:
        return self.stage == 1

    def force_learned_stage(self):
        """Dùng cho ablation NoTwoTimescale."""
        self.stage = 1
        self.alpha_slow = self.alpha_slow_base
        self.accel_remaining = 0

    def step_episode(self):
        self.episode += 1

        if self.stage == 0 and self.episode >= self.k0_warmup:
            self.stage = 1
            self.alpha_slow = self.alpha_slow_base

        if self.accel_remaining > 0:
            self.accel_remaining -= 1

            if self.accel_remaining == 0:
                self.alpha_slow = self.alpha_slow_base

    def _base_freq(self) -> int:
        return max(1, int(round(1.0 / max(1e-8, self.alpha_slow_ratio))))

    def _accel_freq(self) -> int:
        r = max(1e-8, self.alpha_slow_ratio * max(1.0, self.accel_factor))
        return max(1, int(round(1.0 / r)))

    def should_update_graph(self) -> bool:
        """
        True nếu tới lượt train proxy + cập nhật belief/core.

        KHÔNG dùng hàm này để quyết định có thu thập replay hay không —
        replay phải thu thập MỌI episode, kể cả Stage 0.
        """
        if self.stage == 0:
            return False

        freq = (
            self._accel_freq() if self.accel_remaining > 0 else self._base_freq()
        )

        return (self.episode % freq) == 0

    # ------------------------------------------------------------------
    # Phát hiện shift
    # ------------------------------------------------------------------

    def _in_refractory(self) -> bool:
        if self.last_trigger_episode is None:
            return False

        return (self.episode - self.last_trigger_episode) < self.refractory

    def evaluate_drift(
        self,
        probe_z: float = 0.0,
        matrix_z: float = 0.0,
        belief_modules: Optional[Dict] = None,
        drift_detector=None,
    ) -> Dict:
        """
        Gọi một lần mỗi episode sau khi đã đo hai cò súng.

        Args:
            probe_z: z-score từ DriftDetector.residual_z_score()
            matrix_z: z-score từ MatrixDriftDetector.z_score()
            belief_modules: {ego_id: BayesLightBeliefState} để bơm phồng.
            drift_detector: để hẹn lịch chụp ảnh lại sau khi thích nghi.

        Returns:
            dict trạng thái, có khoá "fired".
        """
        out = {
            "episode": int(self.episode),
            "probe_z": float(probe_z),
            "matrix_z": float(matrix_z),
            "fired": False,
            "reason": None,
            "n_inflated": 0,
        }

        if self.stage == 0:
            out["reason"] = "warmup"
            return out

        if self._in_refractory():
            out["reason"] = "refractory"
            return out

        hit_probe = float(probe_z) > self.z_threshold
        hit_matrix = float(matrix_z) > self.z_threshold

        fired = (
            (hit_probe and hit_matrix) if self.require_both
            else (hit_probe or hit_matrix)
        )

        if not fired:
            out["reason"] = "below_threshold"
            return out

        # ---- BẮN --------------------------------------------------------
        self.trigger_count += 1
        self.last_trigger_episode = self.episode

        # (a) tăng tốc cập nhật cấu trúc
        self.alpha_slow = self.alpha_slow_base * self.accel_factor
        self.accel_remaining = self.accel_duration

        # (b) bơm phồng bất định -> core tự co lại -> thận trọng
        n_inflated = 0

        if belief_modules is not None:
            for mod in belief_modules.values():
                if hasattr(mod, "inflate_uncertainty"):
                    st = mod.inflate_uncertainty(
                        factor=self.inflation_factor,
                        t_reset=self.inflation_t_reset,
                    )
                    n_inflated += int(st["n_pairs_inflated"])

        # (c) hẹn lịch chụp ảnh lại cho probe, nếu không nó báo động mãi
        if drift_detector is not None and hasattr(drift_detector, "notify_trigger"):
            drift_detector.notify_trigger(self.episode)

        out.update({
            "fired": True,
            "reason": (
                "both" if (hit_probe and hit_matrix)
                else ("probe" if hit_probe else "matrix")
            ),
            "n_inflated": int(n_inflated),
        })

        self.trigger_log.append(dict(out))

        return out

    # ------------------------------------------------------------------
    # Tương thích ngược API v1
    # ------------------------------------------------------------------

    def _residual_z_score(self, residual: float) -> float:
        """Quy residual thô về z-score dựa trên cửa sổ gần đây (tự quản,
        tách biệt khỏi DriftDetector/MatrixDriftDetector)."""
        self._residual_history.append(float(residual))

        if len(self._residual_history) > 500:
            del self._residual_history[:-500]

        h = self._residual_history

        if len(h) < 5:
            return 0.0

        recent = np.asarray(h[-self._residual_window:], dtype=np.float64)
        base = recent[:-1]

        if base.size < 3:
            return 0.0

        mu, sd = float(np.mean(base)), float(np.std(base))

        if sd < 1e-9:
            return 0.0

        return float((h[-1] - mu) / sd)

    def record_structural_residual(self, residual: float) -> bool:
        """
        Wrapper tương thích ngược cho runner cũ (`final_runner.py`,
        `baseline_runner.py`) vốn gọi API v1 với MỘT residual thô thay vì
        hai z-score độc lập của `evaluate_drift()`.

        Tự tính z-score nội bộ từ residual rồi coi đó là cò súng "probe";
        không có cò súng "matrix" (matrix_z=0.0). Nếu runner có sẵn
        DriftDetector/MatrixDriftDetector thật, nên gọi thẳng
        `evaluate_drift()` thay vì hàm này.

        Returns:
            True nếu trigger bắn (tương đương `triggered` ở API v1).
        """
        z = self._residual_z_score(residual)
        out = self.evaluate_drift(probe_z=z, matrix_z=0.0)

        return bool(out["fired"])

    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        return {
            "episode": int(self.episode),
            "stage": int(self.stage),
            "alpha_fast": float(self.alpha_fast),
            "alpha_slow": float(self.alpha_slow),
            "accel_remaining": int(self.accel_remaining),
            "trigger_count": int(self.trigger_count),
            "last_trigger_episode": self.last_trigger_episode,
            "in_refractory": bool(self._in_refractory()),
            "z_threshold": float(self.z_threshold),
        }


