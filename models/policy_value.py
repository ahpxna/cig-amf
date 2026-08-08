import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    """
    Policy-value network nhận đúng 4 input của CIG-AMF:

        obs_i
        core_summary_i
        peripheral_summary_i
        belief_summary_i

    Không đổi interface so với bản cũ.

    Method-level role:
    - obs_i: observation riêng của ego-agent i.
    - core_summary_i: explicit pair-relational core summary Z_i.
    - peripheral_summary_i: peripheral approximation M_i.
    - belief_summary_i: Bayes-light structural belief summary B_i.

    Output:
        logits: policy logits over discrete action space.
        value: scalar value estimate cho actor-critic update.
    """

    def __init__(
        self,
        obs_dim,
        core_dim,
        peripheral_dim,
        belief_dim,
        action_dim,
        hidden=160,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.core_dim = int(core_dim)
        self.peripheral_dim = int(peripheral_dim)
        self.belief_dim = int(belief_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)

        in_dim = (
            self.obs_dim
            + self.core_dim
            + self.peripheral_dim
            + self.belief_dim
        )

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )

        self.actor = nn.Linear(self.hidden, self.action_dim)
        self.critic = nn.Linear(self.hidden, 1)

    def forward(self, obs, core_summary, peripheral_summary, belief_summary):
        # 1. Gom tất cả input vào 1 list
        inputs = [
            obs,
            core_summary,
            peripheral_summary,
            belief_summary,
        ]

        # 2. Lấy số lượng batch/agents (ví dụ: B = 24) từ obs
        B = obs.shape[0]

        # 3. [FIX-CRIT-1b] Bản cũ .expand(B,-1) MỌI tensor có shape[0]==1 một
        # cách âm thầm. Đó chính là thứ đã che giấu bug "peripheral memory của
        # agent cuối bị nhân bản cho cả 24 agent" trong final_runner suốt nhiều
        # lần debug: input sai shape không crash, chỉ lặng lẽ cho kết quả sai.
        # Giữ lại expand cho trường hợp HỢP LỆ duy nhất (B == 1, gọi single-ego),
        # còn lệch shape khi B > 1 thì phải nổ ngay tại chỗ.
        names = ("obs", "core_summary", "peripheral_summary", "belief_summary")
        expanded_inputs = []
        for name, t in zip(names, inputs):
            if t.dim() > 1 and t.shape[0] == 1 and B > 1:
                raise ValueError(
                    f"PolicyValueNet.forward: '{name}' có batch=1 trong khi "
                    f"obs có batch={B}. Đây gần như luôn là lỗi wiring "
                    f"(một summary bị tính cho 1 ego rồi dùng cho tất cả), "
                    f"không phải broadcast hợp lệ."
                )
            expanded_inputs.append(t)

        # 4. Cat lại an toàn
        x = torch.cat(expanded_inputs, dim=-1)

        h = self.backbone(x)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        return logits, value