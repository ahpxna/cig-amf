import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefItemEncoder(nn.Module):
    def __init__(self, item_dim: int = 9, hidden: int = 24):
        super().__init__()

        self.item_dim = int(item_dim)
        self.hidden = int(hidden)

        self.net = nn.Sequential(
            nn.Linear(self.item_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BeliefSummaryBuilder(nn.Module):
    """
    B_i = [global_stats || top-k structural profile || pooled context]

    item format:
        0: mu_bar
        1: sigma_bar
        2: p_core
        3: in_core
        4: in_seed_core
        5: rel_row
        6: rel_col
        7: zone_diff
        8: pair_latent_norm

    Bám paper:
    - Belief summary không chỉ là binary core set.
    - Nó chứa global statistics, top-k structural profile, pooled context embedding.
    - Top-k structural profile phải có tín hiệu ở Stage 0, không được chết khi mu=0/sigma=1.
    """

    def __init__(
        self,
        top_k: int = 4,
        pooled_hidden: int = 24,
        out_dim: int = 64,
        priority_mu_floor: float = 0.02,
    ):
        super().__init__()

        self.top_k = int(top_k)
        self.item_dim = 9
        self.pooled_hidden = int(pooled_hidden)
        self.priority_mu_floor = float(priority_mu_floor)

        self.item_enc = BeliefItemEncoder(
            item_dim=self.item_dim,
            hidden=self.pooled_hidden,
        )

        self.attn = nn.Linear(self.pooled_hidden, 1)

        global_dim = 9
        topk_dim = self.top_k * self.item_dim
        pooled_dim = self.pooled_hidden

        self.pre_out_dim = global_dim + topk_dim + pooled_dim

        self.out_proj = nn.Sequential(
            nn.Linear(self.pre_out_dim, int(out_dim)),
            nn.ReLU(),
        )

        self.out_dim = int(out_dim)

    def build_item(self, ego_id, j, env, belief_state, pair_latent_norm=None):
        ego_id = int(ego_id)
        j = int(j)

        pi = env.positions[ego_id]
        pj = env.positions[j]

        b = belief_state[j]

        grid_den = max(1, int(env.grid_size))
        zone_den = max(1, int(env.n_zones) - 1)

        return np.array(
            [
                float(b["mu_bar"]),
                float(b["sigma_bar"]),
                float(b["p_core"]),
                float(b["in_core"]),
                float(b["in_seed_core"]),
                float((pj[0] - pi[0]) / grid_den),
                float((pj[1] - pi[1]) / grid_den),
                float((env.agent_zone[j] - env.agent_zone[ego_id]) / zone_den),
                float(0.0 if pair_latent_norm is None else pair_latent_norm),
            ],
            dtype=np.float32,
        )

    def _device(self):
        return next(self.parameters()).device

    def _std(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 1:
            return torch.zeros((), dtype=x.dtype, device=x.device)

        return torch.std(x, unbiased=False)

    def _to_tensor(self, items_np):
        device = self._device()

        if items_np is None:
            return torch.zeros(
                0,
                self.item_dim,
                dtype=torch.float32,
                device=device,
            )

        if isinstance(items_np, torch.Tensor):
            items = items_np.to(device=device, dtype=torch.float32)
        elif isinstance(items_np, np.ndarray):
            items = torch.from_numpy(items_np).to(
                device=device,
                dtype=torch.float32,
            )
        else:
            items = torch.tensor(
                np.asarray(items_np, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )

        if items.dim() == 1:
            items = items.unsqueeze(0)

        return items

    def _empty_summary(self):
        device = self._device()

        x = torch.zeros(
            1,
            self.pre_out_dim,
            dtype=torch.float32,
            device=device,
        )

        return self.out_proj(x).squeeze(0)

    def forward(self, items_np) -> torch.Tensor:
        if items_np is None or len(items_np) == 0:
            return self._empty_summary()

        items = self._to_tensor(items_np)

        if items.shape[0] == 0:
            return self._empty_summary()

        mu_col = items[:, 0]
        sigma_col = items[:, 1]
        p_col = items[:, 2]
        in_core_col = items[:, 3]
        in_seed_col = items[:, 4]

        global_stats = torch.stack(
            [
                torch.mean(mu_col),
                self._std(mu_col),
                torch.mean(sigma_col),
                self._std(sigma_col),
                torch.mean(p_col),
                self._std(p_col),
                torch.mean(in_core_col),
                torch.mean(in_seed_col),
                torch.tensor(
                    float(items.shape[0]),
                    dtype=items.dtype,
                    device=items.device,
                ),
            ],
            dim=0,
        ).unsqueeze(0)

        # Sửa Stage 0 priority:
        # Bản yếu:
        #     confidence = clamp(1 - sigma, 0, 1)
        #     priority = |mu| * (0.5 + p) * confidence
        #
        # Khi sigma=1 và mu=0 thì priority=0 cho mọi item, top-k bị quyết định
        # gần như theo index. Bản này dùng confidence mềm và mu_floor.
        sigma_safe = torch.clamp(sigma_col, min=0.0)
        confidence = 1.0 / (1.0 + sigma_safe)

        priority = (
            (torch.abs(mu_col) + self.priority_mu_floor)
            * (0.5 + p_col)
            * confidence
        )

        top_idx = torch.argsort(priority, descending=True)[: self.top_k]
        top_items = items[top_idx]

        if top_items.shape[0] < self.top_k:
            pad = torch.zeros(
                self.top_k - top_items.shape[0],
                self.item_dim,
                dtype=items.dtype,
                device=items.device,
            )
            top_items = torch.cat([top_items, pad], dim=0)

        top_flat = top_items.reshape(1, -1)

        h = self.item_enc(items)

        attn_logits = self.attn(h).squeeze(-1)
        attn_weight = F.softmax(attn_logits, dim=0)
        pooled = torch.sum(h * attn_weight[:, None], dim=0, keepdim=True)

        cat = torch.cat(
            [
                global_stats,
                top_flat,
                pooled,
            ],
            dim=-1,
        )

        return self.out_proj(cat).squeeze(0)