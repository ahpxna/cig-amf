import numpy as np
import torch
import torch.nn as nn

from envs.causal_adapter import compact_relation_features, resolve_env_adapter
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
    """B_i = [global_stats || top-k structural profile || pooled context]

item format (the relation suffix is adapter-owned and opaque to the model):
    0: capacity_bar
    1: sigma_capacity_bar
    2: g_score = C - kappa*sigma_C
    3: in_core
    4: in_seed_core (diagnostic membership)
    5--7: compact adapter relation features
    8: pair_latent_norm

Follow paper:
- Belief summary is not just a binary core set.
- It contains global statistics, top-k structural profile, pooled context embedding.
- Top-k structural profile must have signal in Stage 0, and must not die when mu=0/sigma=1."""

    def __init__(
        self,
        top_k: int = 4,
        pooled_hidden: int = 24,
        out_dim: int = 64,
        priority_mu_floor: float = 0.0,
    ):
        super().__init__()

        self.top_k = int(top_k)
        self.item_dim = 9
        self.pooled_hidden = int(pooled_hidden)
        self.priority_mu_floor = float(priority_mu_floor)
        if abs(self.priority_mu_floor) > 1e-12:
            raise ValueError(
                "belief summaries use the literal G=C-kappa*sigma score; "
                "priority_mu_floor is not part of the final contract"
            )

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

        b = belief_state[j]
        relation = compact_relation_features(
            resolve_env_adapter(env), ego_id, j, width=3
        )

        return np.array(
            [
                float(b["capacity_bar"]),
                float(b["sigma_capacity_bar"]),
                float(
                    b.get(
                        "g_score",
                        float(b.get("capacity_bar", 0.0))
                        - float(b.get("sigma_capacity_bar", b.get("sigma_bar", 0.0))),
                    )
                ),
                float(b["in_core"]),
                float(b["in_seed_core"]),
                *[float(value) for value in relation],
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
        g_col = items[:, 2]
        in_core_col = items[:, 3]
        in_seed_col = items[:, 4]

        global_stats = torch.stack(
            [
                torch.mean(mu_col),
                self._std(mu_col),
                torch.mean(sigma_col),
                self._std(sigma_col),
                torch.mean(g_col),
                self._std(g_col),
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

        # Top-k follows the same lower-confidence-bound score G used by the
        # structural allocator.  This keeps the public belief summary aligned
        # with the paper's C-only selection semantics instead of reintroducing
        # a probability-like p_core proxy.
        priority = g_col

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
