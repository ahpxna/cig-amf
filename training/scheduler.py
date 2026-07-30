class TwoTimescaleScheduler:
    """
    Scheduler cho CIG-AMF:

    Stage 0:
        - seeded-core warm-up.
        - policy/value, pair-relational module, peripheral encoder học trên fast timescale.
        - replay/proxy buffer được collect.
        - proxy/belief/core learned update chưa takeover.

    Stage 1:
        - proxy ensemble train định kỳ.
        - Bayes-light belief update định kỳ.
        - core/peripheral partition dùng learned belief + hysteresis.
        - residual EWMA/CUSUM trigger có thể tăng tạm thời tần suất structural update.

    Bám paper:
        - fast process: policy, value, pair latent, shadow latent.
        - slow process: proxy, belief, graph/core update.
        - trigger dùng proxy residual chứ không dùng critic loss.

    Quy tắc quan trọng:
        should_update_graph() chỉ quyết định train proxy + update belief/core.
        Nó không quyết định collect replay.
        Replay/proxy buffer phải được collect every episode ở runner.
    """

    def __init__(
        self,
        k0_warmup=20,
        alpha_fast=1e-3,
        alpha_slow_ratio=0.05,
        accel_factor=4.0,
        accel_duration=8,
        ewma_alpha=0.10,
        cusum_threshold=0.03,
        cusum_drift=0.003,
    ):
        self.k0_warmup = int(k0_warmup)

        self.alpha_fast = float(alpha_fast)
        self.alpha_slow_ratio = float(alpha_slow_ratio)
        self.alpha_slow_base = self.alpha_fast * self.alpha_slow_ratio
        self.alpha_slow = 0.0

        self.accel_factor = float(accel_factor)
        self.accel_duration = int(accel_duration)

        self.ewma_alpha = float(ewma_alpha)
        self.cusum_threshold = float(cusum_threshold)
        self.cusum_drift = float(cusum_drift)

        self.episode = 0
        self.stage = 0

        self.residual_ewma = None
        self.cusum_score = 0.0

        self.accel_remaining = 0
        self.trigger_count = 0

    def in_warmup(self):
        return self.stage == 0

    def in_learned_stage(self):
        return self.stage == 1

    def force_learned_stage(self):
        """
        Dùng cho NoTwoTimescale ablation.

        Nếu NoTwoTimescale bị giữ ở Stage 0 thì weak prior sẽ reset mãi,
        làm ablation không còn là kiểm tra scheduler nữa.

        Sau khi gọi hàm này:
        - stage = 1
        - alpha_slow dùng base slow step.
        - không dùng acceleration pending từ trước.
        """
        self.stage = 1
        self.alpha_slow = self.alpha_slow_base
        self.accel_remaining = 0

    def step_episode(self):
        """
        Gọi đúng một lần sau mỗi episode.

        Chuyển Stage 0 -> Stage 1 khi số episode đã đạt k0_warmup.
        Nếu đang trong accelerated structural update window thì giảm counter.
        """
        self.episode += 1

        if self.stage == 0 and self.episode >= self.k0_warmup:
            self.stage = 1
            self.alpha_slow = self.alpha_slow_base

        if self.accel_remaining > 0:
            self.accel_remaining -= 1

            if self.accel_remaining == 0:
                self.alpha_slow = self.alpha_slow_base

    def _base_update_frequency(self):
        """
        Convert slow ratio thành update frequency.

        Ví dụ:
            alpha_slow_ratio = 0.05 -> update khoảng mỗi 20 episodes.
            alpha_slow_ratio = 0.10 -> update khoảng mỗi 10 episodes.
        """
        ratio = max(1e-8, self.alpha_slow_ratio)
        return max(1, int(round(1.0 / ratio)))

    def _accelerated_update_frequency(self):
        """
        Frequency khi residual trigger bật acceleration.

        Ví dụ:
            slow_ratio = 0.05, accel_factor = 4
            effective ratio = 0.20
            update khoảng mỗi 5 episodes.
        """
        ratio = max(
            1e-8,
            self.alpha_slow_ratio * max(1.0, self.accel_factor),
        )
        return max(1, int(round(1.0 / ratio)))

    def should_update_graph(self):
        """
        True nếu tới lượt train proxy + update belief/core.

        Không được dùng hàm này để quyết định có collect replay hay không.
        Replay/proxy buffer phải collect every episode ở runner.
        """
        if self.stage == 0:
            return False

        if self.accel_remaining > 0:
            freq = self._accelerated_update_frequency()
        else:
            freq = self._base_update_frequency()

        return (self.episode % freq) == 0

    def record_structural_residual(self, residual):
        """
        EWMA/CUSUM trên proxy residual.

        E_t = (1 - lambda_E) E_{t-1} + lambda_E residual_t
        S_t = max(0, S_{t-1} + residual_t - E_{t-1} - drift)

        Args:
            residual:
                proxy holdout residual hoặc residual summary mới nhất.

        Return:
            bool trigger.
        """
        residual = float(residual)

        if self.residual_ewma is None:
            self.residual_ewma = residual
            return False

        prev = self.residual_ewma

        self.residual_ewma = (
            (1.0 - self.ewma_alpha) * self.residual_ewma
            + self.ewma_alpha * residual
        )

        excess = residual - prev - self.cusum_drift
        self.cusum_score = max(0.0, self.cusum_score + excess)

        if self.stage == 1 and self.cusum_score > self.cusum_threshold:
            self.alpha_slow = self.alpha_slow_base * self.accel_factor
            self.accel_remaining = self.accel_duration
            self.cusum_score = 0.0
            self.trigger_count += 1
            return True

        return False

    def get_status(self):
        return {
            "episode": int(self.episode),
            "stage": int(self.stage),
            "alpha_fast": float(self.alpha_fast),
            "alpha_slow": float(self.alpha_slow),
            "alpha_slow_ratio": float(self.alpha_slow_ratio),
            "residual_ewma": (
                None
                if self.residual_ewma is None
                else float(self.residual_ewma)
            ),
            "cusum_score": float(self.cusum_score),
            "trigger_count": int(self.trigger_count),
            "accel_remaining": int(self.accel_remaining),
        }