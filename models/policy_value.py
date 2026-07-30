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
        x = torch.cat(
            [
                obs,
                core_summary,
                peripheral_summary,
                belief_summary,
            ],
            dim=-1,
        )

        h = self.backbone(x)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        return logits, value