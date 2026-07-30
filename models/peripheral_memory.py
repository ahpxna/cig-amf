import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeripheralMultiMemory(nn.Module):
    """
    Peripheral multi-memory encoder cho CIG-AMF.

    Input chính:
        periph_items: ndarray hoặc tensor shape [N_p, item_dim]

    item format mặc định:
        0: action_j
        1: mu_bar
        2: sigma_bar
        3: p_core
        4: in_prev_core
        5: rel_row
        6: rel_col
        7: zone_diff
        8: distance_norm

    Output:
        M_{P_i}: vector shape [out_dim]

    Sửa quan trọng:
    - Stage 0 thường mu_bar=0 và sigma_bar=1.
    - Nếu importance = p_core * abs(mu) / sigma thì beta = 0, memory chết.
    - Bản này dùng:
        beta = (beta_floor + p_core) * (abs(mu) + mu_floor) * confidence
        confidence = 1 / (1 + sigma)
    - Đồng thời trộn learned weighted memory với uniform memory để giữ peripheral signal.
    """

    def __init__(
        self,
        action_dim,
        num_slots=4,
        memory_dim=32,
        out_dim=64,
        item_hidden=48,
        item_dim=9,
        mu_floor=0.02,
        beta_floor=0.05,
        uniform_mix=0.25,
        eps=1e-6,
    ):
        super().__init__()

        self.action_dim = int(action_dim)
        self.num_slots = int(num_slots)
        self.memory_dim = int(memory_dim)
        self.out_dim = int(out_dim)
        self.item_hidden = int(item_hidden)
        self.item_dim = int(item_dim)

        self.mu_floor = float(mu_floor)
        self.beta_floor = float(beta_floor)
        self.uniform_mix = float(uniform_mix)
        self.eps = float(eps)

        # item gồm action index + belief features + relation features.
        # action được one-hot rồi concat với phần non-action.
        self.non_action_dim = self.item_dim - 1
        self.encoder_in_dim = self.action_dim + self.non_action_dim

        self.item_encoder = nn.Sequential(
            nn.Linear(self.encoder_in_dim, self.item_hidden),
            nn.ReLU(),
            nn.Linear(self.item_hidden, self.memory_dim),
            nn.ReLU(),
        )

        self.slot_router = nn.Sequential(
            nn.Linear(self.encoder_in_dim, self.item_hidden),
            nn.ReLU(),
            nn.Linear(self.item_hidden, self.num_slots),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(self.num_slots * self.memory_dim, self.out_dim),
            nn.ReLU(),
        )

    def _device(self):
        return next(self.parameters()).device

    def _one_hot_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long().clamp(min=0, max=self.action_dim - 1)
        return F.one_hot(
            actions,
            num_classes=self.action_dim,
        ).to(dtype=torch.float32)

    def _normalise_inputs(self, periph_items):
        device = self._device()

        if periph_items is None:
            return torch.zeros(
                0,
                self.item_dim,
                dtype=torch.float32,
                device=device,
            )

        if isinstance(periph_items, np.ndarray):
            x = torch.from_numpy(periph_items).to(
                device=device,
                dtype=torch.float32,
            )
        elif isinstance(periph_items, torch.Tensor):
            x = periph_items.to(device=device, dtype=torch.float32)
        else:
            x = torch.tensor(
                np.asarray(periph_items, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.numel() == 0:
            return torch.zeros(
                0,
                self.item_dim,
                dtype=torch.float32,
                device=device,
            )

        if x.shape[-1] != self.item_dim:
            raise ValueError(
                f"PeripheralMultiMemory expected item_dim={self.item_dim}, "
                f"got last dim={x.shape[-1]}"
            )

        return x

    def build_inputs(
        self,
        ego_id,
        peripheral_ids,
        env,
        belief_state,
        prev_core_set=None,
    ):
        """
        Build peripheral item matrix cho một ego-agent.

        Args:
            ego_id:
                ego index.
            peripheral_ids:
                list/set neighbour ids thuộc P_i.
            env:
                main env có positions, agent_zone, get_action_dim(), last_actions.
            belief_state:
                dict {j: state dict} từ BayesLightBeliefState.get_state_dict().
            prev_core_set:
                set core trước đó, dùng làm feature in_prev_core.

        Return:
            np.ndarray shape [len(peripheral_ids), 9]
        """
        ego_id = int(ego_id)
        prev_core_set = set() if prev_core_set is None else set(prev_core_set)

        ids = [
            int(j)
            for j in list(peripheral_ids)
            if int(j) != ego_id
        ]

        if len(ids) == 0:
            return np.zeros((0, self.item_dim), dtype=np.float32)

        pi = env.positions[ego_id]
        grid_den = max(1, int(env.grid_size))
        zone_den = max(1, int(env.n_zones) - 1)

        if hasattr(env, "last_actions"):
            last_actions = env.last_actions
        else:
            last_actions = [0 for _ in range(int(env.n_agents))]

        rows = []

        for j in ids:
            pj = env.positions[j]
            b = belief_state[j]

            action_j = int(last_actions[j])
            action_j = int(np.clip(action_j, 0, self.action_dim - 1))

            rel_row = float((pj[0] - pi[0]) / grid_den)
            rel_col = float((pj[1] - pi[1]) / grid_den)
            dist_norm = (
                float(abs(pj[0] - pi[0]) + abs(pj[1] - pi[1]))
                / grid_den
            )
            zone_diff = float(
                (env.agent_zone[j] - env.agent_zone[ego_id])
                / zone_den
            )

            rows.append(
                [
                    float(action_j),
                    float(b["mu_bar"]),
                    float(b["sigma_bar"]),
                    float(b["p_core"]),
                    float(j in prev_core_set),
                    rel_row,
                    rel_col,
                    zone_diff,
                    dist_norm,
                ]
            )

        return np.asarray(rows, dtype=np.float32)

    def _prepare_encoder_input(self, items: torch.Tensor):
        action_col = items[:, 0].long().clamp(min=0, max=self.action_dim - 1)
        action_oh = self._one_hot_actions(action_col)
        rest = items[:, 1:].to(dtype=torch.float32)
        return torch.cat([action_oh, rest], dim=-1)

    def _importance_beta(self, items: torch.Tensor):
        mu = items[:, 1]
        sigma = torch.clamp(items[:, 2], min=0.0)
        p_core = torch.clamp(items[:, 3], min=0.0, max=1.0)

        mu_abs = torch.abs(mu)
        confidence = 1.0 / (1.0 + sigma + self.eps)

        beta = (
            (self.beta_floor + p_core)
            * (mu_abs + self.mu_floor)
            * confidence
        )

        beta = torch.clamp(beta, min=self.eps)

        return beta

    def forward(self, periph_items):
        items = self._normalise_inputs(periph_items)
        device = self._device()

        if items.shape[0] == 0:
            x = torch.zeros(
                1,
                self.num_slots * self.memory_dim,
                dtype=torch.float32,
                device=device,
            )
            return self.out_proj(x).squeeze(0)

        enc_in = self._prepare_encoder_input(items)

        h = self.item_encoder(enc_in)
        slot_logits = self.slot_router(enc_in)
        slot_probs = F.softmax(slot_logits, dim=-1)

        beta = self._importance_beta(items)

        memories = []

        for q in range(self.num_slots):
            assign_q = slot_probs[:, q]
            weighted = assign_q * beta

            denom = torch.sum(weighted) + self.eps
            learned_mem = torch.sum(h * weighted[:, None], dim=0) / denom

            uniform_denom = torch.sum(assign_q) + self.eps
            uniform_mem = torch.sum(h * assign_q[:, None], dim=0) / uniform_denom

            mix = float(np.clip(self.uniform_mix, 0.0, 1.0))
            mem_q = (1.0 - mix) * learned_mem + mix * uniform_mem

            memories.append(mem_q)

        flat = torch.cat(memories, dim=-1).unsqueeze(0)
        return self.out_proj(flat).squeeze(0)