from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairRelationalEncoder(nn.Module):
    """
    GRUCell encoder cho pair-specific relational latent z_ij.

    Bám paper:
        z_ij^t = f_z(z_ij^{t-1}, o_i^t, o_j^t, a_i^t, a_j^t, xi_ij^t)

    Input vector:
        [obs_i, obs_j, onehot(a_i), onehot(a_j), rel_features]

    State:
        z_ij
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        rel_feat_dim,
        hidden_dim,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.rel_feat_dim = int(rel_feat_dim)
        self.hidden_dim = int(hidden_dim)

        self.input_dim = (
            self.obs_dim
            + self.obs_dim
            + self.action_dim
            + self.action_dim
            + self.rel_feat_dim
        )

        self.gru = nn.GRUCell(self.input_dim, self.hidden_dim)

    def forward(self, x, h_prev):
        return self.gru(x, h_prev)


class ShadowPairEncoder(nn.Module):
    """
    Lightweight shadow state s_ij cho mọi directed pair.

    Bám paper:
        s_ij^t = f_s(s_ij^{t-1}, o_j^t, a_j^t, xi_ij^t)

    Shadow state rẻ hơn full z_ij.
    Khi j được promote vào core, z_ij được warm-start từ s_ij.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        rel_feat_dim,
        shadow_dim,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.rel_feat_dim = int(rel_feat_dim)
        self.shadow_dim = int(shadow_dim)

        self.input_dim = (
            self.obs_dim
            + self.action_dim
            + self.rel_feat_dim
        )

        self.gru = nn.GRUCell(self.input_dim, self.shadow_dim)

    def forward(self, x, s_prev):
        return self.gru(x, s_prev)


class PairRelationalModule:
    """
    Pair-specific relational module.

    Maintains:
        full_states[(i, j)]   = z_ij
        shadow_states[(i, j)] = s_ij

    Required public methods:
        clone_full_states_np()
        clone_shadow_states_np()
        warm_start_if_promoted()
        get_core_summary()
        get_pair_latent()
        step_population()
        add_bc_transition()
        train_bc()

    Method consistency:
    - z_ij^t is updated using action-selection-time context:
          o_i^t, o_j^t, a_i^t, a_j^t, Delta_ij^t
    - Runner should restore env to snapshot-before-step before calling step_population().
    - For behavioural cloning, runner should pass h_prev_snapshot and, if available,
      s_prev_snapshot captured before step_population().
    - train_bc() trains the recurrent encoders one-step through:
          z_next = full_encoder(x_full_t, z_prev)
          s_next = shadow_encoder(x_shadow_t, s_prev)
      then predicts a_j^{t+1}.
    """

    def __init__(
        self,
        n_agents,
        obs_dim,
        action_dim,
        hidden_dim=64,
        shadow_dim=16,
        rel_feat_dim=6,
        lr=1e-3,
        device="cpu",
        bc_buffer_size=200000,
        max_bc_buffer=None,
        grad_clip=1.0,
        shadow_loss_weight=0.35,
    ):
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.shadow_dim = int(shadow_dim)
        self.rel_feat_dim = int(rel_feat_dim)
        self.lr = float(lr)
        self.device = device
        self.grad_clip = float(grad_clip)
        self.shadow_loss_weight = float(shadow_loss_weight)

        # Backward compatibility: vài bản cũ gọi max_bc_buffer.
        if max_bc_buffer is not None:
            bc_buffer_size = int(max_bc_buffer)
        self.bc_buffer_size = int(bc_buffer_size)

        self.full_encoder = PairRelationalEncoder(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            rel_feat_dim=self.rel_feat_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.shadow_encoder = ShadowPairEncoder(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            rel_feat_dim=self.rel_feat_dim,
            shadow_dim=self.shadow_dim,
        ).to(self.device)

        self.shadow_to_full = nn.Linear(
            self.shadow_dim,
            self.hidden_dim,
        ).to(self.device)

        # Head dùng chung trên full latent space.
        # Full path: z_next -> bc_head -> a_j^{t+1}
        # Shadow path: shadow_to_full(s_next) -> bc_head -> a_j^{t+1}
        self.bc_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        ).to(self.device)

        params = (
            list(self.full_encoder.parameters())
            + list(self.shadow_encoder.parameters())
            + list(self.shadow_to_full.parameters())
            + list(self.bc_head.parameters())
        )

        self.optim = torch.optim.Adam(params, lr=self.lr)

        self.full_states = {}
        self.shadow_states = {}

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue

                pair = (int(i), int(j))

                self.full_states[pair] = torch.zeros(
                    1,
                    self.hidden_dim,
                    dtype=torch.float32,
                    device=self.device,
                )

                self.shadow_states[pair] = torch.zeros(
                    1,
                    self.shadow_dim,
                    dtype=torch.float32,
                    device=self.device,
                )

        self.bc_buffer = deque(maxlen=self.bc_buffer_size)

        self.last_bc_loss = 0.0
        self.last_full_bc_loss = 0.0
        self.last_shadow_bc_loss = 0.0
        self.last_bc_batch_count = 0

    # ============================================================
    # Basic tensor helpers
    # ============================================================

    def _one_hot_action_np(self, action):
        a = int(action)
        a = max(0, min(self.action_dim - 1, a))

        x = np.zeros((self.action_dim,), dtype=np.float32)
        x[a] = 1.0

        return x

    def _one_hot_action_tensor(self, actions):
        if isinstance(actions, torch.Tensor):
            a = actions.to(device=self.device, dtype=torch.long)
        else:
            a = torch.tensor(actions, dtype=torch.long, device=self.device)

        if a.dim() == 0:
            a = a.unsqueeze(0)

        a = torch.clamp(a, min=0, max=self.action_dim - 1)

        return F.one_hot(a, num_classes=self.action_dim).to(dtype=torch.float32)

    def _obs_np(self, obs):
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)

        if arr.shape[0] != self.obs_dim:
            raise ValueError(
                f"Expected obs_dim={self.obs_dim}, got {arr.shape[0]}"
            )

        return arr.astype(np.float32)

    def _to_tensor_2d(self, arr, expected_dim=None):
        t = torch.tensor(
            np.asarray(arr, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        if t.dim() == 1:
            t = t.unsqueeze(0)

        if expected_dim is not None and t.shape[-1] != int(expected_dim):
            raise ValueError(
                f"Expected dim={expected_dim}, got {t.shape[-1]}"
            )

        return t

    def _get_from_container(self, obj, idx):
        if isinstance(obj, dict):
            return obj[int(idx)]
        return obj[int(idx)]

    # ============================================================
    # Relational feature construction
    # ============================================================

    def _rel_features_np(self, ego_id, neighbor_id, env):
        """
        Build xi_ij.

        rel_feat_dim mặc định 6:
            0 rel_row
            1 rel_col
            2 manhattan distance normalised
            3 same_zone
            4 zone_diff normalised
            5 same_role / role_match indicator if available

        Nếu rel_feat_dim khác 6, pad/truncate để không crash.
        """
        ego_id = int(ego_id)
        neighbor_id = int(neighbor_id)

        grid_den = max(1, int(getattr(env, "grid_size", 1)))
        zone_den = max(1, int(getattr(env, "n_zones", 1)) - 1)

        pi = env.positions[ego_id]
        pj = env.positions[neighbor_id]

        rel_row = float((pj[0] - pi[0]) / grid_den)
        rel_col = float((pj[1] - pi[1]) / grid_den)

        dist = float(abs(pj[0] - pi[0]) + abs(pj[1] - pi[1]))
        dist_norm = dist / grid_den

        zi = int(env.agent_zone[ego_id])
        zj = int(env.agent_zone[neighbor_id])

        same_zone = 1.0 if zi == zj else 0.0
        zone_diff = float((zj - zi) / zone_den)

        same_role = 0.0
        if hasattr(env, "agent_role"):
            try:
                same_role = 1.0 if env.agent_role[ego_id] == env.agent_role[neighbor_id] else 0.0
            except Exception:
                same_role = 0.0

        # [P-8 FINAL DEBUG] Định danh HỢP LỆ: agent id chuẩn hóa (nhãn tùy ý,
        # không mang thông tin cấu trúc). Feature thứ 7 chỉ có hiệu lực khi
        # khởi tạo với rel_feat_dim=7; mặc định 6 sẽ truncate => no-op, không
        # phá checkpoint/test cũ. TUYỆT ĐỐI KHÔNG thêm ROLE ID thật của
        # neighbor: role là diagnostic ground truth của paper (Exp 4 chấm
        # "recovered-role accuracy") — đưa vào input là rò rỉ nhãn, vô hiệu
        # hóa claim H3/RQ3 và làm bẩn H1.
        agent_id_norm = float(neighbor_id) / float(max(1, getattr(env, "n_agents", 1)))

        feats = [
            rel_row,
            rel_col,
            dist_norm,
            same_zone,
            zone_diff,
            same_role,
            agent_id_norm,
        ]

        if len(feats) < self.rel_feat_dim:
            feats = feats + [0.0 for _ in range(self.rel_feat_dim - len(feats))]

        if len(feats) > self.rel_feat_dim:
            feats = feats[: self.rel_feat_dim]

        return np.asarray(feats, dtype=np.float32)

    def _build_full_input_np(
        self,
        obs_i,
        obs_j,
        action_i,
        action_j,
        rel_feat,
    ):
        obs_i_np = self._obs_np(obs_i)
        obs_j_np = self._obs_np(obs_j)

        ai = self._one_hot_action_np(action_i)
        aj = self._one_hot_action_np(action_j)

        rel = np.asarray(rel_feat, dtype=np.float32).reshape(-1)

        if rel.shape[0] != self.rel_feat_dim:
            raise ValueError(
                f"Expected rel_feat_dim={self.rel_feat_dim}, got {rel.shape[0]}"
            )

        x = np.concatenate(
            [
                obs_i_np,
                obs_j_np,
                ai,
                aj,
                rel,
            ],
            axis=0,
        ).astype(np.float32)

        return x

    def _build_shadow_input_np(
        self,
        obs_j,
        action_j,
        rel_feat,
    ):
        obs_j_np = self._obs_np(obs_j)
        aj = self._one_hot_action_np(action_j)

        rel = np.asarray(rel_feat, dtype=np.float32).reshape(-1)

        if rel.shape[0] != self.rel_feat_dim:
            raise ValueError(
                f"Expected rel_feat_dim={self.rel_feat_dim}, got {rel.shape[0]}"
            )

        x = np.concatenate(
            [
                obs_j_np,
                aj,
                rel,
            ],
            axis=0,
        ).astype(np.float32)

        return x

    # ============================================================
    # State cloning for runner timing correctness
    # ============================================================

    def clone_full_states_np(self):
        """
        Clone full z_ij states before step_population().

        Runner dùng snapshot này để tạo BC transition đúng thời điểm:
            context at t -> predict a_j at t+1
        """
        out = {}

        for pair, state in self.full_states.items():
            out[pair] = (
                state.detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
                .copy()
            )

        return out

    def clone_shadow_states_np(self):
        """
        Clone shadow s_ij states before step_population().

        Không bắt buộc với runner cũ. Nếu runner chưa truyền snapshot này,
        add_bc_transition() sẽ fallback sang current shadow state.
        """
        out = {}

        for pair, state in self.shadow_states.items():
            out[pair] = (
                state.detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
                .copy()
            )

        return out

    # ============================================================
    # Online latent updates
    # ============================================================

    def step_population(self, obs_all, actions, env):
        """
        Update z_ij và s_ij cho mọi directed pair ở timestep hiện tại.

        Yêu cầu runner:
            env phải đang ở snapshot trước step.
            obs_all và actions là observation/action tại cùng timestep t.

        Update này là online state filtering, không giữ graph gradient qua thời gian.
        Encoder weights vẫn được train bằng train_bc() trên one-step samples.
        """
        with torch.no_grad():
            for ego in range(self.n_agents):
                obs_i = self._get_from_container(obs_all, ego)
                action_i = self._get_from_container(actions, ego)

                for j in range(self.n_agents):
                    if j == ego:
                        continue

                    pair = (int(ego), int(j))

                    obs_j = self._get_from_container(obs_all, j)
                    action_j = self._get_from_container(actions, j)
                    rel_feat = self._rel_features_np(ego, j, env)

                    x_full_np = self._build_full_input_np(
                        obs_i=obs_i,
                        obs_j=obs_j,
                        action_i=action_i,
                        action_j=action_j,
                        rel_feat=rel_feat,
                    )

                    x_shadow_np = self._build_shadow_input_np(
                        obs_j=obs_j,
                        action_j=action_j,
                        rel_feat=rel_feat,
                    )

                    x_full = self._to_tensor_2d(
                        x_full_np,
                        expected_dim=self.full_encoder.input_dim,
                    )

                    x_shadow = self._to_tensor_2d(
                        x_shadow_np,
                        expected_dim=self.shadow_encoder.input_dim,
                    )

                    h_prev = self.full_states[pair]
                    s_prev = self.shadow_states[pair]

                    h_next = self.full_encoder(x_full, h_prev)
                    s_next = self.shadow_encoder(x_shadow, s_prev)

                    self.full_states[pair] = h_next.detach()
                    self.shadow_states[pair] = s_next.detach()

    # ============================================================
    # Warm-start and summaries
    # ============================================================

    def warm_start_if_promoted(self, ego_id, promoted_ids):
        """
        Khi j được promote vào core C_i:
            z_ij <- W_proj s_ij

        Return:
            số pair được warm-start.
        """
        ego_id = int(ego_id)
        count = 0

        if promoted_ids is None:
            return 0

        with torch.no_grad():
            for j in promoted_ids:
                j = int(j)

                if j == ego_id:
                    continue

                pair = (ego_id, j)

                if pair not in self.shadow_states:
                    continue

                s = self.shadow_states[pair]
                z = self.shadow_to_full(s)

                self.full_states[pair] = z.detach()
                count += 1

        return int(count)

    def get_pair_latent(self, ego_id, neighbor_id):
        pair = (int(ego_id), int(neighbor_id))

        if pair not in self.full_states:
            return np.zeros((self.hidden_dim,), dtype=np.float32)

        return (
            self.full_states[pair]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32)
        )

    def get_shadow_latent(self, ego_id, neighbor_id):
        pair = (int(ego_id), int(neighbor_id))

        if pair not in self.shadow_states:
            return np.zeros((self.shadow_dim,), dtype=np.float32)

        return (
            self.shadow_states[pair]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32)
        )

    def get_core_summary(self, ego_id, core_set):
        """
        Mean pooling over pair-specific z_ij for j in C_i.

        Bám paper:
            Z_i = Pool({z_ij : j in C_i})
        """
        ego_id = int(ego_id)

        if core_set is None:
            return np.zeros((self.hidden_dim,), dtype=np.float32)

        ids = [
            int(j)
            for j in list(core_set)
            if int(j) != ego_id and (ego_id, int(j)) in self.full_states
        ]

        if len(ids) == 0:
            return np.zeros((self.hidden_dim,), dtype=np.float32)

        vals = [
            self.get_pair_latent(ego_id, j)
            for j in ids
        ]

        return np.mean(
            np.stack(vals, axis=0),
            axis=0,
        ).astype(np.float32)

    def get_core_summary_excluding_all(self, ego_id, core_set):
        """
        Tối ưu cho runner phần sau.

        Return:
            dict {j: Z_i^{-j}} cho mọi neighbour j != ego.

        Nếu j không thuộc core:
            Z_i^{-j} = mean({z_ik : k in C_i})

        Nếu j thuộc core:
            Z_i^{-j} = mean({z_ik : k in C_i, k != j})

        Hàm này giúp final_runner không phải gọi get_core_summary()
        lặp lại N lần.
        """
        ego_id = int(ego_id)

        all_neighbors = [
            j for j in range(self.n_agents)
            if j != ego_id
        ]

        core_ids = [
            int(j)
            for j in list(core_set or [])
            if int(j) != ego_id and (ego_id, int(j)) in self.full_states
        ]

        out = {}

        if len(core_ids) == 0:
            zero = np.zeros((self.hidden_dim,), dtype=np.float32)

            for j in all_neighbors:
                out[int(j)] = zero.copy()

            return out

        core_latents = {
            j: self.get_pair_latent(ego_id, j)
            for j in core_ids
        }

        sum_core = np.sum(
            np.stack(list(core_latents.values()), axis=0),
            axis=0,
        ).astype(np.float32)

        mean_core = (sum_core / float(len(core_ids))).astype(np.float32)

        for j in all_neighbors:
            if j not in core_latents:
                out[int(j)] = mean_core.copy()
                continue

            if len(core_ids) <= 1:
                out[int(j)] = np.zeros((self.hidden_dim,), dtype=np.float32)
            else:
                excl = (
                    (sum_core - core_latents[j])
                    / float(len(core_ids) - 1)
                ).astype(np.float32)

                out[int(j)] = excl

        return out

    # ============================================================
    # Behavioural cloning replay
    # ============================================================

    def add_bc_transition(
        self,
        observations,
        actions,
        next_actions,
        env,
        h_prev_snapshot=None,
        s_prev_snapshot=None,
    ):
        """
        Add supervised one-step behavioural prediction samples.

        Target:
            context at t -> neighbour action a_j at t+1

        Stored fields:
            x_full_t:
                [o_i^t, o_j^t, onehot(a_i^t), onehot(a_j^t), xi_ij^t]
            x_shadow_t:
                [o_j^t, onehot(a_j^t), xi_ij^t]
            h_prev:
                z_ij^{t-1}, captured before online update when available.
            s_prev:
                s_ij^{t-1}, captured before online update when available.
            target_action:
                a_j^{t+1}

        Critical fix:
            train_bc() will run encoder forward again:
                z_t = full_encoder(x_full_t, h_prev)
                s_t = shadow_encoder(x_shadow_t, s_prev)
            so gradient updates full_encoder and shadow_encoder, not only bc_head.
        """
        for ego in range(self.n_agents):
            obs_i = self._get_from_container(observations, ego)
            action_i = self._get_from_container(actions, ego)

            for j in range(self.n_agents):
                if j == ego:
                    continue

                pair = (int(ego), int(j))

                obs_j = self._get_from_container(observations, j)
                action_j = self._get_from_container(actions, j)
                target_action_j = int(self._get_from_container(next_actions, j))
                target_action_j = max(0, min(self.action_dim - 1, target_action_j))

                rel_feat = self._rel_features_np(ego, j, env)

                x_full = self._build_full_input_np(
                    obs_i=obs_i,
                    obs_j=obs_j,
                    action_i=action_i,
                    action_j=action_j,
                    rel_feat=rel_feat,
                )

                x_shadow = self._build_shadow_input_np(
                    obs_j=obs_j,
                    action_j=action_j,
                    rel_feat=rel_feat,
                )

                if h_prev_snapshot is not None and pair in h_prev_snapshot:
                    h_prev = np.asarray(
                        h_prev_snapshot[pair],
                        dtype=np.float32,
                    ).reshape(-1)
                else:
                    h_prev = (
                        self.full_states[pair]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                        .astype(np.float32)
                    )

                if s_prev_snapshot is not None and pair in s_prev_snapshot:
                    s_prev = np.asarray(
                        s_prev_snapshot[pair],
                        dtype=np.float32,
                    ).reshape(-1)
                else:
                    s_prev = (
                        self.shadow_states[pair]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                        .astype(np.float32)
                    )

                if h_prev.shape[0] != self.hidden_dim:
                    raise ValueError(
                        f"h_prev dim mismatch for pair={pair}: "
                        f"expected {self.hidden_dim}, got {h_prev.shape[0]}"
                    )

                if s_prev.shape[0] != self.shadow_dim:
                    raise ValueError(
                        f"s_prev dim mismatch for pair={pair}: "
                        f"expected {self.shadow_dim}, got {s_prev.shape[0]}"
                    )

                self.bc_buffer.append(
                    {
                        "ego_id": int(ego),
                        "neighbor_id": int(j),
                        "x_full": x_full.astype(np.float32),
                        "x_shadow": x_shadow.astype(np.float32),
                        "h_prev": h_prev.astype(np.float32),
                        "s_prev": s_prev.astype(np.float32),
                        "target_action": int(target_action_j),
                    }
                )

    def _sample_bc_batch(self, batch_size):
        if len(self.bc_buffer) == 0:
            return []

        n = min(int(batch_size), len(self.bc_buffer))
        return random.sample(list(self.bc_buffer), n)

    def _bc_batch_to_tensors(self, batch):
        x_full = np.stack(
            [b["x_full"] for b in batch],
            axis=0,
        ).astype(np.float32)

        x_shadow = np.stack(
            [b["x_shadow"] for b in batch],
            axis=0,
        ).astype(np.float32)

        h_prev = np.stack(
            [b["h_prev"] for b in batch],
            axis=0,
        ).astype(np.float32)

        s_prev = np.stack(
            [b["s_prev"] for b in batch],
            axis=0,
        ).astype(np.float32)

        target = np.asarray(
            [b["target_action"] for b in batch],
            dtype=np.int64,
        )

        x_full_t = torch.tensor(
            x_full,
            dtype=torch.float32,
            device=self.device,
        )

        x_shadow_t = torch.tensor(
            x_shadow,
            dtype=torch.float32,
            device=self.device,
        )

        h_prev_t = torch.tensor(
            h_prev,
            dtype=torch.float32,
            device=self.device,
        )

        s_prev_t = torch.tensor(
            s_prev,
            dtype=torch.float32,
            device=self.device,
        )

        target_t = torch.tensor(
            target,
            dtype=torch.long,
            device=self.device,
        )

        return x_full_t, x_shadow_t, h_prev_t, s_prev_t, target_t

    def train_bc(
        self,
        n_steps=1,
        batch_size=256,
        heads=None,
        heads_optim=None,
        w_contrastive=0.3,
        w_influence=1.0,
        w_target_fn=None,
    ):
        """
        Train auxiliary behavioural prediction objective.

        Bám paper:
            L_z = -log p(a_j^{t+1} | z_ij^t)

        Sửa critical:
            full_encoder và shadow_encoder được train qua one-step recurrent update.
            Không chỉ train một prediction head trên detached h_prev.

        Loss:
            full_loss:
                z_t = full_encoder(x_full_t, z_prev)
                logits = bc_head(z_t)
                CE(logits, a_j^{t+1})

            shadow_loss:
                s_t = shadow_encoder(x_shadow_t, s_prev)
                z_shadow = shadow_to_full(s_t)
                logits_shadow = bc_head(z_shadow)
                CE(logits_shadow, a_j^{t+1})

            total = full_loss + shadow_loss_weight * shadow_loss

        [ego_conditioned_latent.py — vá pair-specificity, xem docstring
        file đó] Nếu truyền `heads` (EgoConditionedHeads), cắm thêm:

            total += w_influence * L_influence + w_contrastive * L_contrastive

        vào CÙNG loss, backward CÙNG lượt với full_loss/shadow_loss. Đây là
        điểm mấu chốt: L_influence/L_contrastive PHẢI tính trên `z_next`
        (còn gradient, đầu ra sống của full_encoder ở batch này), KHÔNG
        PHẢI trên `get_pair_latent()` (đã .detach()). Nếu dùng bản detach,
        heads học ra thông tin sẵn có trong z nhưng full_encoder không hề
        bị ép phải NHÉT thêm thông tin về ego vào z -- z_ij vẫn hội tụ về
        global opponent model y như bug gốc, chỉ là heads "diễn" cho có.

        ego_id/neighbor_id của batch lấy trực tiếp từ bc_buffer (đã có sẵn
        từ add_bc_transition) -- KHÔNG gom theo ego khi gọi hàm này ở
        runner, vì một batch bc_buffer tự nhiên trộn nhiều ego với nhau
        (bc_buffer là buffer chung của mọi pair), nên neg_mask của
        contrastive_loss (cùng j khác ego) tự nhiên không rỗng.

        w_target_fn: callable(ego_id, neighbor_id) -> float, dùng làm nhãn
        w_ij cho influence_loss. Không có nhãn causal riêng cho bc_buffer
        nên dùng belief.debiased_mu(j) hiện tại của runner làm proxy nhãn
        (đây chính là ước lượng w_ij tốt nhất đang có tại thời điểm gọi).
        None = bỏ qua influence_loss (chỉ dùng contrastive).
        """
        self.last_bc_batch_count = 0
        self.last_heads_loss = 0.0

        if len(self.bc_buffer) == 0:
            self.last_bc_loss = 0.0
            self.last_full_bc_loss = 0.0
            self.last_shadow_bc_loss = 0.0
            return 0.0

        n_steps = int(max(0, n_steps))
        batch_size = int(max(1, batch_size))

        if n_steps == 0:
            return 0.0

        losses = []
        full_losses = []
        shadow_losses = []

        self.full_encoder.train()
        self.shadow_encoder.train()
        self.shadow_to_full.train()
        self.bc_head.train()

        for _ in range(n_steps):
            batch = self._sample_bc_batch(batch_size)

            if len(batch) == 0:
                continue

            (
                x_full_t,
                x_shadow_t,
                h_prev_t,
                s_prev_t,
                target_t,
            ) = self._bc_batch_to_tensors(batch)

            z_next = self.full_encoder(x_full_t, h_prev_t)
            logits_full = self.bc_head(z_next)
            full_loss = F.cross_entropy(logits_full, target_t)

            s_next = self.shadow_encoder(x_shadow_t, s_prev_t)
            z_from_shadow = self.shadow_to_full(s_next)
            logits_shadow = self.bc_head(z_from_shadow)
            shadow_loss = F.cross_entropy(logits_shadow, target_t)

            loss = full_loss + self.shadow_loss_weight * shadow_loss

            heads_loss_val = None
            if heads is not None:
                ego_ids_batch = [int(s["ego_id"]) for s in batch]
                nb_ids_batch = [int(s["neighbor_id"]) for s in batch]

                ego_t = torch.tensor(
                    ego_ids_batch, dtype=torch.long, device=self.device
                )
                nb_t = torch.tensor(
                    nb_ids_batch, dtype=torch.long, device=self.device
                )

                # [E2] dùng z_next CÒN GRADIENT -> ép full_encoder thật sự
                # phải nhét thông tin ego vào z_ij (xem docstring hàm này).
                con_loss = heads.contrastive_loss(z_next, ego_t, nb_t)

                if w_target_fn is not None:
                    w_vals = [
                        float(w_target_fn(ego_ids_batch[k], nb_ids_batch[k]))
                        for k in range(len(batch))
                    ]
                    w_t = torch.tensor(
                        w_vals, dtype=torch.float32, device=self.device
                    )
                    inf_loss = heads.influence_loss(z_next, w_t)
                else:
                    inf_loss = torch.zeros(
                        (), dtype=torch.float32, device=self.device
                    )

                heads_loss_val = w_influence * inf_loss + w_contrastive * con_loss
                loss = loss + heads_loss_val

            self.optim.zero_grad()
            if heads is not None and heads_optim is not None:
                heads_optim.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.full_encoder.parameters())
                + list(self.shadow_encoder.parameters())
                + list(self.shadow_to_full.parameters())
                + list(self.bc_head.parameters()),
                self.grad_clip,
            )

            if heads is not None:
                torch.nn.utils.clip_grad_norm_(heads.parameters(), self.grad_clip)

            self.optim.step()

            if heads is not None and heads_optim is not None:
                heads_optim.step()

            losses.append(float(loss.detach().cpu().item()))
            full_losses.append(float(full_loss.detach().cpu().item()))
            shadow_losses.append(float(shadow_loss.detach().cpu().item()))

            if heads_loss_val is not None:
                self.last_heads_loss = float(heads_loss_val.detach().cpu().item())

            self.last_bc_batch_count += 1

        if len(losses) == 0:
            self.last_bc_loss = 0.0
            self.last_full_bc_loss = 0.0
            self.last_shadow_bc_loss = 0.0
            return 0.0

        self.last_bc_loss = float(np.mean(losses))
        self.last_full_bc_loss = float(np.mean(full_losses))
        self.last_shadow_bc_loss = float(np.mean(shadow_losses))

        return float(self.last_bc_loss)

    # ============================================================
    # Diagnostics
    # ============================================================

    def get_bc_buffer_size(self):
        return int(len(self.bc_buffer))

    def get_last_bc_loss(self):
        return float(self.last_bc_loss)

    def get_last_full_bc_loss(self):
        return float(self.last_full_bc_loss)

    def get_last_shadow_bc_loss(self):
        return float(self.last_shadow_bc_loss)

    def get_last_bc_batch_count(self):
        return int(self.last_bc_batch_count)

    def get_debug_stats(self):
        full_norms = []
        shadow_norms = []

        for pair in self.full_states:
            full_norms.append(
                float(torch.norm(self.full_states[pair]).detach().cpu().item())
            )
            shadow_norms.append(
                float(torch.norm(self.shadow_states[pair]).detach().cpu().item())
            )

        return {
            "n_agents": int(self.n_agents),
            "n_pairs": int(len(self.full_states)),
            "hidden_dim": int(self.hidden_dim),
            "shadow_dim": int(self.shadow_dim),
            "bc_buffer_size": int(len(self.bc_buffer)),
            "last_bc_loss": float(self.last_bc_loss),
            "last_full_bc_loss": float(self.last_full_bc_loss),
            "last_shadow_bc_loss": float(self.last_shadow_bc_loss),
            "last_bc_batch_count": int(self.last_bc_batch_count),
            "mean_full_state_norm": (
                float(np.mean(full_norms))
                if len(full_norms) > 0
                else 0.0
            ),
            "mean_shadow_state_norm": (
                float(np.mean(shadow_norms))
                if len(shadow_norms) > 0
                else 0.0
            ),
        }