from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.causal_adapter import resolve_env_adapter


class PairRelationalEncoder(nn.Module):
    """
    GRUCell encoder cho pair-specific relational latent z_ij.

    Paper correspondence:
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
    Lightweight shadow state s_ij for every directed pair.

    Paper correspondence:
        s_ij^t = f_s(s_ij^{t-1}, o_j^t, a_j^t, xi_ij^t)

    The shadow state is cheaper than the full z_ij state. When j is promoted
    into the core, z_ij is warm-started from s_ij.
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
        state_mode="recurrent",
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
        self.state_mode = str(state_mode).strip().lower()
        if self.state_mode not in {"recurrent", "feedforward", "aggregate"}:
            raise ValueError(
                "state_mode must be recurrent, feedforward, or aggregate"
            )

        # Backward compatibility: some older versions use max_bc_buffer.
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

        # Shared head over the full latent space.
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
        self.cd_norm_mean = np.zeros(2, dtype=np.float32)
        self.cd_norm_std = np.ones(2, dtype=np.float32)
        self.cd_normalization_frozen = False

    def fit_cd_normalization(self, min_samples=32):
        """Fit and freeze C/D scaling from the pre-confirmatory replay."""
        values = [
            np.asarray(sample["cd_target"], dtype=np.float32)
            for sample in self.bc_buffer
            if sample.get("cd_target") is not None
        ]
        if len(values) < int(min_samples):
            return False
        table = np.stack(values, axis=0)
        self.cd_norm_mean = np.mean(table, axis=0).astype(np.float32)
        std = np.std(table, axis=0).astype(np.float32)
        self.cd_norm_std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        self.cd_normalization_frozen = True
        return True

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

        The default rel_feat_dim is 6:
            0 rel_row
            1 rel_col
            2 manhattan distance normalised
            3 same_zone
            4 zone_diff normalised
            5 same_role / role_match indicator if available

        Pad or truncate when rel_feat_dim differs from 6.
        """
        ego_id = int(ego_id)
        neighbor_id = int(neighbor_id)

        adapter = resolve_env_adapter(env)
        pair = np.asarray(
            adapter.pair_features(ego_id, neighbor_id), dtype=np.float32
        )
        if pair.size < 5:
            raise ValueError("adapter pair_features must expose five base channels")
        rel_row, rel_col, dist_norm, same_zone, zone_diff = pair[:5]
        same_role = 0.0
        ego_pair = np.asarray(
            adapter.pair_features(neighbor_id, ego_id), dtype=np.float32
        )
        if pair.size > 5 and ego_pair.size == pair.size:
            same_role = float(
                np.argmax(pair[5:]) == np.argmax(ego_pair[5:])
                and np.max(pair[5:]) > 0.0
                and np.max(ego_pair[5:]) > 0.0
            )

        # Optional identity feature: normalized agent ID, an arbitrary label
        # carrying no structural information. The seventh feature is active
        # only with rel_feat_dim=7; the default value 6 truncates it to a no-op.
        # The public role is already present in OmniArena's neighbour
        # observation, so it is not duplicated as a separate relational
        # channel here.
        agent_id_norm = float(neighbor_id) / float(max(1, adapter.n_agents))

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

        The runner uses this snapshot to create a BC transition at the correct time:
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

        Optional for legacy runners. If no snapshot is supplied,
        add_bc_transition() falls back to the current shadow state.
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
        Update z_ij and s_ij for every directed pair at the current timestep.

        Runner requirements:
            env must be at the pre-step snapshot.
            obs_all and actions must describe the same timestep t.

        This update performs online state filtering without retaining a
        gradient graph through time. Encoder weights are still trained by
        train_bc() on one-step samples.
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
                    if self.state_mode == "feedforward":
                        h_prev = torch.zeros_like(h_prev)
                    s_prev = self.shadow_states[pair]

                    h_next = self.full_encoder(x_full, h_prev)
                    s_next = self.shadow_encoder(x_shadow, s_prev)

                    self.full_states[pair] = h_next.detach()
                    self.shadow_states[pair] = s_next.detach()

            if self.state_mode == "aggregate":
                # Remove ego identity from the relational state while
                # retaining a neighbour-specific aggregate baseline.
                for neighbor in range(self.n_agents):
                    pairs = [
                        (ego, neighbor)
                        for ego in range(self.n_agents)
                        if ego != neighbor
                    ]
                    shared = torch.mean(
                        torch.cat([self.full_states[pair] for pair in pairs], dim=0),
                        dim=0,
                        keepdim=True,
                    )
                    for pair in pairs:
                        self.full_states[pair] = shared.detach().clone()

    # ============================================================
    # Warm-start and summaries
    # ============================================================

    def warm_start_if_promoted(self, ego_id, promoted_ids):
        """
        When j is promoted into core C_i:
            z_ij <- W_proj s_ij

        Return:
            Number of warm-started pairs.
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

        Paper correspondence:
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
        Optimization for the downstream runner.

        Return:
            Dictionary {j: Z_i^{-j}} for every neighbour j != ego.

        If j is outside the core:
            Z_i^{-j} = mean({z_ik : k in C_i})

        If j is in the core:
            Z_i^{-j} = mean({z_ik : k in C_i, k != j})

        This avoids N repeated get_core_summary() calls in final_runner.
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
        cd_target_fn=None,
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
                        "cd_target": (
                            None
                            if cd_target_fn is None
                            else np.asarray(
                                cd_target_fn(int(ego), int(j)), dtype=np.float32
                            ).reshape(2)
                        ),
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
        cd_target_fn=None,
    ):
        """
        Train auxiliary behavioural prediction objective.

        Paper correspondence:
            L_z = -log p(a_j^{t+1} | z_ij^t)

        Critical correction:
            full_encoder and shadow_encoder are trained through a one-step
            recurrent update, rather than training only a prediction head on
            detached h_prev.

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

        [ego_conditioned_latent.py — pair-specificity correction; see that
        file's docstring] When `heads` (EgoConditionedHeads) is provided, add:

            total += w_influence * L_CD + w_contrastive * L_contrastive

        to the same loss and backpropagate it together with full_loss and
        shadow_loss. Crucially, L_influence/L_contrastive must be computed on
        `z_next`, the live full_encoder output retaining gradients for this
        batch, not on detached `get_pair_latent()`. With the detached version,
        the heads can read information already present in z, but full_encoder
        is never forced to encode additional ego information. z_ij then still
        converges to a global opponent model as in the original defect, while
        the heads only create the appearance of pair specificity.

        Batch ego_id/neighbor_id values come directly from bc_buffer. A shared
        pair buffer mixes egos. Same-neighbour/different-ego examples become
        hard negatives only when their C/D targets differ materially.

        cd_target_fn: callable(ego_id, neighbor_id) -> ``[C,D]``.  C is the
        current debiased structural-capacity belief and D comes from the fast
        response signature tracker.  None disables the C/D head while keeping
        the action-prediction objective.
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

            recurrent_state = (
                torch.zeros_like(h_prev_t)
                if self.state_mode == "feedforward"
                else h_prev_t
            )
            z_next = self.full_encoder(x_full_t, recurrent_state)
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

                # [E2] Use z_next with gradients intact so full_encoder must
                # encode ego information in z_ij; see this method's docstring.
                labelled = [sample.get("cd_target") is not None for sample in batch]
                if any(labelled):
                    labelled_idx = [index for index, valid in enumerate(labelled) if valid]
                    cd_vals = [batch[index]["cd_target"] for index in labelled_idx]
                    cd_vals = (
                        np.asarray(cd_vals, dtype=np.float32)
                        - self.cd_norm_mean.reshape(1, 2)
                    ) / self.cd_norm_std.reshape(1, 2)
                    cd_t = torch.tensor(
                        np.asarray(cd_vals, dtype=np.float32),
                        dtype=torch.float32, device=self.device,
                    )
                    selected_z = z_next[labelled_idx]
                    selected_ego = ego_t[labelled_idx]
                    selected_nb = nb_t[labelled_idx]
                    inf_loss = heads.influence_loss(selected_z, cd_t)
                    con_loss = heads.contrastive_loss(
                        selected_z, selected_ego, selected_nb, cd_targets=cd_t
                    )
                else:
                    inf_loss = torch.zeros(
                        (), dtype=torch.float32, device=self.device
                    )
                    con_loss = torch.zeros(
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
