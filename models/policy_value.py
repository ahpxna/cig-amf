import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    """Policy-value network receives the correct 4 inputs of CIG-AMF:

        obs_i
        core_summary_i
        peripheral_summary_i
        belief_summary_i

    Maintains the same interface as before.

    Method-level role:
    - obs_i: observation specific to ego-agent i.
    - core_summary_i: explicit pair-relational core summary Z_i.
    - peripheral_summary_i: peripheral approximation M_i.
    - belief_summary_i: Bayes-light structural belief summary B_i.

    Output:
        logits: policy logits over discrete action space.
        value: scalar value estimate for actor-critic update."""

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
        # 1. Gather all inputs into one list
        inputs = [
            obs,
            core_summary,
            peripheral_summary,
            belief_summary,
        ]

        # 2. Get the batch/agent count (e.g. B = 24) from obs
        B = obs.shape[0]

        # 3. [FIX-CRIT-1b] The old .expand(B,-1) applies to every tensor with shape[0]==1
        # in a silent way. That is exactly what concealed the bug "peripheral memory of
        # the last agent was duplicated for all 24 agents" in final_runner for many
        # debugging sessions: input with wrong shape did not crash, only silently gave wrong results.
        # Keep expand for the only valid case (B == 1, called single-ego),
        # while shape mismatch when B > 1 must raise an error immediately.
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

        # 4. Safe truncation
        x = torch.cat(expanded_inputs, dim=-1)

        h = self.backbone(x)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        return logits, value
