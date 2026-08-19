"""
structural_proxy.py — Local counterfactual proxy.

FOUR V1 DEFECTS CORRECTED HERE

[L1] THE ESTIMATOR DISCARDED SIGN.
     v1 applied abs(alt_preds - base_preds), forcing mu >= 0. This forced
     p_core to use |mu_bar|, made blockers indistinguishable from helpers,
     and could not match the environment's signed oracle, making Exp. 3
     impossible. The current estimator preserves sign and exposes four
     effect_mode variants for ablation.

[L2] THE PLUG-IN ESTIMATE INHERITED REWARD-MODEL BIAS.
     v1 used w = f(a') - f(a), so biased f produced biased w, while
     non-random a_j made f learn from confounded data. The current estimator
     is doubly robust. MARL supplies exact pi_j because the policy is trained
     within the system, so exact propensity keeps DR unbiased even if f is
     misspecified.

[L3] THE ENSEMBLE WAS EFFECTIVELY IDENTICAL.
     All three v1 members trained on the same batches in the same order and
     converged to nearly the same function, producing sigma = 0.000. Current
     members have independent bootstrap masks, batches, and initializations.
     Their complete forward/backward/update runs as one torch.func.vmap GPU
     operation without a Python loop over models.

[L4] A SINGLE HORIZON PROVIDED NO LATENCY DIMENSION.
     A multi-horizon head predicts R^(1), ..., R^(H) jointly.

GPU OPTIMIZATION (torch.func.vmap ensemble)

The original n_ensemble independent modules caused sequential GPU launches
for identical shapes; every .item()/.cpu() also forced CPU/GPU synchronization.
The correction vectorizes the ensemble dimension:
    1. stack_module_state combines weights into a tensor tree with leading E.
    2. functional_call with vmap evaluates all members in one operation.
    3. One Adam updates the stack while retaining independent elementwise
       moments and preventing cross-member gradient leakage.
    4. Gradient clipping uses a separate norm per member; a global norm would
       let one exploding member clip every other member.
    5. n_ensemble forwards, backwards, and updates become one of each over E.

`self.buffer` intentionally remains a deque of Python dictionaries because
drift_probe.py reads that structure directly. Sampling avoids the former
O(n_ensemble * buffer_size) Python scan: 800,000 operations for a 200k buffer
and four members. Each member instead receives a fixed NumPy permutation mask
computed once at initialization. It identifies approximately bootstrap_ratio
of buffer ranks visible to that member; C-level filtering plus weighted
random.choices oversamples interventions. Per-call mask redraws are forbidden:
[BB1] in GPU_OPTIMIZATION_CONTRACT.md explains that they eventually expose
nearly the entire buffer to every member and reproduce v1's collapse.

BACKWARD COMPATIBILITY

Runner-facing signatures remain compatible: add_sample only adds optional
parameters; train_step, score_batch, and score_pair remain unchanged;
score_batch still returns (mu_arr, sigma_arr); score_batch_full returns the
complete influence-signature dictionary; self.buffer remains deque[dict].
"""

import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call, stack_module_state, vmap
    _HAS_TORCH_FUNC = True
except ImportError:  # Rare torch <2.0 fallback; remain operational.
    _HAS_TORCH_FUNC = False


# =============================================================================
# Proxy network
# =============================================================================

class LocalCounterfactualProxyNet(nn.Module):
    """
    One ensemble member.

    Input matching the paper's conditioning set in Eq. 5:
        obs_i, a_i, a_j, Z_i^{-j}, M_i^{-j}, B_i

    Output:
        [B, n_horizons] predictions of R_i^(1), ..., R_i^(H)

    Multi-horizon rationale:
        Neighbour influence can be delayed. A blocker acts immediately at
        h=1, whereas relay/signaller benefits can appear only at h=3. A single
        aggregate R^(H) makes the two indistinguishable. Horizon separation
        was the signature's sixth dimension before the later 5D revision.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        core_dim: int,
        periph_dim: int,
        belief_dim: int,
        hidden: int = 160,
        n_horizons: int = 8,
        use_belief_input: bool = False,
        dropout: float = 0.0,
        pair_feat_dim: int = 0,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.core_dim = int(core_dim)
        self.periph_dim = int(periph_dim)
        self.belief_dim = int(belief_dim)
        self.hidden = int(hidden)
        self.n_horizons = int(n_horizons)

        # ---------------------------------------------------------------
        # [H6] Break the belief -> proxy -> belief feedback loop.
        #
        # B_i derives from this proxy's own w_hat. Feeding B_i back lets the
        # proxy self-confirm and creates an architectural confounder. Belief
        # input is disabled by default.
        # ---------------------------------------------------------------
        self.use_belief_input = bool(use_belief_input)

        # -------------------------------------------------------------------
        # [FIX-X1] x_ij completes the Eq. 7 -> Eq. 8 refactor.
        #
        # The previous version completed only half the refactor: a_j was
        # removed and Eq. 8's multi-head output added, but x_ij was omitted.
        # At fixed s for ego i, j then affected input only through Z_i^{-j}
        # and M_i^{-j}. Outside the core, Z_i^{-j}=Z_i exactly, while omitting
        # one of roughly 20 items barely changes M_i^{-j}. Thus ŵ_ij was nearly
        # constant across j and supplied no rank signal. This exactly matches
        # the eight-seed H1 Spearman results (0.003, 0.138, -0.123, -0.027),
        # whose bootstrap intervals all contained zero.
        #
        # pair_feat_dim = 0 preserves legacy behaviour.
        # -------------------------------------------------------------------
        self.pair_feat_dim = int(pair_feat_dim)

        self.in_dim = (
            self.obs_dim
            + self.action_dim   # a_i one-hot
            # a_j is no longer an input; see the multi-head output below.
            + self.pair_feat_dim   # x_ij (Eq 8)
            + self.core_dim
            + self.periph_dim
            + (self.belief_dim if self.use_belief_input else 0)
        )

        layers = [
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))

        layers += [
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))

        # One head for every possible action of j.
        layers.append(nn.Linear(self.hidden, self.action_dim * self.n_horizons))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        obs_i: torch.Tensor,            # [..., B, obs_dim]
        action_i_onehot: torch.Tensor,  # [..., B, action_dim]
        z_core_excl_j: torch.Tensor,    # [..., B, core_dim]
        m_periph_excl_j: torch.Tensor,  # [..., B, periph_dim]
        belief_summary: torch.Tensor,   # [..., B, belief_dim]
        pair_feat: torch.Tensor = None, # [..., B, pair_feat_dim] — x_ij (Eq 8)
    ) -> torch.Tensor:
        """
        a_j is no longer an input. The network predicts every action of j
        jointly; callers use a_j only to gather in train_step/score_batch_full.
        Returns: [..., B, action_dim, n_horizons]
        """
        parts = [
            obs_i,
            action_i_onehot,
        ]

        # [FIX-X1] x_ij must immediately follow a_i in the declared in_dim order.
        if self.pair_feat_dim > 0:
            if pair_feat is None:
                raise ValueError(
                    "StructuralProxyNet được khởi tạo với pair_feat_dim="
                    f"{self.pair_feat_dim} nhưng forward() không nhận pair_feat. "
                    "Thiếu x_ij thì f_theta không phân biệt được neighbour j "
                    "(xem FIX-X1) — không cho phép im lặng bỏ qua."
                )
            parts.append(pair_feat)

        parts += [
            z_core_excl_j,
            m_periph_excl_j,
        ]

        if self.use_belief_input:
            parts.append(belief_summary)

        x = torch.cat(parts, dim=-1)  # [..., in_dim]

        out = self.net(x)  # [..., action_dim * n_horizons]
        return out.view(*out.shape[:-1], self.action_dim, self.n_horizons)


# =============================================================================
# Per-member gradient clipping vectorized over E.
# =============================================================================

def _clip_grad_norm_per_member(
    stacked_params: Dict[str, torch.Tensor],
    max_norm: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Clip the gradient norm independently for each ensemble member.

    Applying torch.nn.utils.clip_grad_norm_ directly to the stacked tensors
    computes one global norm across E and all parameters. One exploding member
    would then clip every member and defeat ensemble independence. This
    function computes and scales each E slice separately. It has no Python
    loop over members, only a small fixed loop over layer parameters.

    Returns:
        [E] gradient norms before clipping for optional diagnostics.
    """
    grads = [p.grad for p in stacked_params.values() if p.grad is not None]

    if len(grads) == 0:
        return torch.zeros(0)

    E = int(grads[0].shape[0])
    device = grads[0].device

    sq_sum = torch.zeros(E, device=device, dtype=grads[0].dtype)

    for g in grads:
        sq_sum = sq_sum + g.reshape(E, -1).pow(2).sum(dim=1)

    norm = torch.sqrt(sq_sum)                                    # [E]
    coef = (float(max_norm) / (norm + eps)).clamp(max=1.0)       # [E]

    for g in grads:
        shape = [E] + [1] * (g.dim() - 1)
        g.mul_(coef.view(*shape))

    return norm


# =============================================================================
# Ensemble
# =============================================================================

class LocalCounterfactualProxyEnsemble:
    """
    Signed, doubly robust, multi-horizon ensemble proxy represented on GPU as
    one tensor with an ensemble dimension, rather than n_ensemble models
    iterated in Python.

    ---------------------------------------------------------------------
    FOUR EFFECT MODES
    ---------------------------------------------------------------------
    "signed_aristocrat"  (DEFAULT; used for beneficial/harmful roles)
        w = f(s, a_j_obs) - E_{a' ~ pi_j}[ f(s, a') ]
        w > 0: j's observed action is better than average for i; j helps
        w < 0: j harms i

    "signed_oracle_matched"  (used for Exp. 3 calibration)
        w = mean_{a in candidates}[ f(s,a) ] - f(s, a_j_obs)
        Exactly matches the environment oracle formula.

    "range"  (Pieroth ICML 2024-style control baseline)
        w = max_a f(s,a) - min_a f(s,a), always >= 0

    "mean_abs"  (v1 form retained for before/after ablation)
        w = mean_{a != a_obs} |f(s,a) - f(s,a_obs)|

    ---------------------------------------------------------------------
    DOUBLY ROBUST
    ---------------------------------------------------------------------
        psi_DR(a) = f_hat(s,a) + (1{a_obs = a} / b_j(a|s)) * (R_obs - f_hat(s,a_obs))

    Either a correct outcome model or a correct propensity is sufficient for
    unbiasedness. MARL provides exact propensity, satisfying one condition.
    Importance weight 1/b is clipped to prevent variance explosion at small b.
    """

    # Four valid modes.
    MODES = (
        "signed_aristocrat",
        "signed_oracle_matched",
        "range",
        "mean_abs",
    )
    MODE_ALIASES = {
        "unsigned_range": "range",
    }

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        core_dim: int,
        periph_dim: int,
        belief_dim: int,
        n_ensemble: int = 4,
        hidden: int = 160,
        lr: float = 1e-3,
        buffer_size: int = 200000,
        device: str = "cpu",
        grad_clip: float = 1.0,
        eps: float = 1e-8,
        n_horizons: int = 8,
        effect_mode: str = "signed_aristocrat",
        use_doubly_robust: bool = True,
        iw_clip: float = 10.0,
        bootstrap_ratio: float = 0.8,
        use_belief_input: bool = False,
        candidate_actions: Optional[List[int]] = None,
        ensemble_dropout: float = 0.0,
        seed: int = 0,
        use_vmap_ensemble: bool = True,
        compile_ensemble: bool = False,
        pair_feat_dim: int = 0,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.core_dim = int(core_dim)
        self.periph_dim = int(periph_dim)
        self.belief_dim = int(belief_dim)
        # [FIX-X1] x_ij from Eq. 8; zero disables it for legacy behaviour.
        self.pair_feat_dim = int(pair_feat_dim)
        self.n_ensemble = int(n_ensemble)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.buffer_size = int(buffer_size)
        self.device = device
        self.grad_clip = float(grad_clip)
        self.eps = float(eps)

        self.n_horizons = int(n_horizons)

        effect_mode = self.MODE_ALIASES.get(str(effect_mode), str(effect_mode))
        if effect_mode not in self.MODES:
            raise ValueError(
                f"effect_mode phải thuộc {self.MODES}, nhận '{effect_mode}'"
            )
        self.effect_mode = str(effect_mode)

        self.use_doubly_robust = bool(use_doubly_robust)
        self.iw_clip = float(iw_clip)
        self.forced_boost = 4.0   # [FIX-HC3b] xem _sample_for_member
        self.bootstrap_ratio = float(np.clip(bootstrap_ratio, 0.1, 1.0))
        self.use_belief_input = bool(use_belief_input)

        self.candidate_actions = (
            list(range(self.action_dim))
            if candidate_actions is None
            else [int(a) for a in candidate_actions]
        )

        # ---------------------------------------------------------------
        # [L3] Genuine ensemble diversity represented as one tensor over E.
        # Diversity comes from independent initializations, bootstrap masks,
        # and batches for each member.
        # ---------------------------------------------------------------
        self.models: List[LocalCounterfactualProxyNet] = []

        for k in range(self.n_ensemble):
            torch.manual_seed(int(seed) * 1000 + k)

            self.models.append(
                LocalCounterfactualProxyNet(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    core_dim=self.core_dim,
                    periph_dim=self.periph_dim,
                    belief_dim=self.belief_dim,
                    pair_feat_dim=self.pair_feat_dim,   # [FIX-X1]
                    hidden=self.hidden,
                    n_horizons=self.n_horizons,
                    use_belief_input=self.use_belief_input,
                    dropout=float(ensemble_dropout),
                ).to(self.device)
            )

        self.use_vmap_ensemble = bool(use_vmap_ensemble) and _HAS_TORCH_FUNC
        self._compile_ensemble_flag = bool(compile_ensemble)

        if self.use_vmap_ensemble:
            self._setup_vmap_ensemble()
        else:
            # torch <2.0 fallback: numerically correct legacy loop without full
            # GPU utilization.
            self.optims = [
                torch.optim.Adam(m.parameters(), lr=self.lr) for m in self.models
            ]

        # Independent RNG per member for independent weighted oversampling.
        self._member_rngs = [
            random.Random(int(seed) * 7919 + k) for k in range(self.n_ensemble)
        ]

        # ---------------------------------------------------------------
        # [BB1 — GPU_OPTIMIZATION_CONTRACT.md] Fixed bootstrap masks; never
        # redraw them on each call.
        #
        # If every train_step draws a new member pool, as the previous version
        # did to avoid Python scans, all members eventually see nearly the full
        # buffer and differ only in individual minibatches. The systematic
        # difference disappears, functions converge, and Eq. 10 sigma
        # collapses to zero as in v1.
        #
        # Correction: each member receives a fixed, independently seeded
        # permutation selecting about bootstrap_ratio of buffer positions.
        # These are ranks rather than samples: deque elements shift over time,
        # but the visible rank set remains fixed. Each member consistently
        # excludes the same fraction, preserving systematic diversity.
        # ---------------------------------------------------------------
        self._member_pool_mask: List[np.ndarray] = []
        keep_n = max(1, int(self.buffer_size * self.bootstrap_ratio))

        for k in range(self.n_ensemble):
            rng_k = np.random.RandomState(int(seed) * 7919 + k)
            perm = rng_k.permutation(self.buffer_size)
            mask = np.zeros(self.buffer_size, dtype=bool)
            mask[perm[:keep_n]] = True
            self._member_pool_mask.append(mask)

        self.buffer = deque(maxlen=self.buffer_size)

        # Diagnostics consumed by legacy runners.
        self.last_train_called = False
        self.last_train_batch_count = 0
        self.latest_residual = 0.0
        self.latest_train_residual = 0.0
        self.latest_holdout_residual = 0.0
        self.latest_loss = 0.0

        # New diagnostics.
        self.latest_ensemble_disagreement = 0.0
        self.latest_dr_correction_magnitude = 0.0
        self.n_interventional_samples = 0

        # [BB3] Per-member losses let T3 compare member 0's gradient scale at
        # E=1 and E=4 without inspecting the autograd graph. None before the
        # first training call.
        self.latest_loss_per_member: Optional[np.ndarray] = None

    # =====================================================================
    # Configure the vmap ensemble path.
    # =====================================================================

    def _setup_vmap_ensemble(self):
        """
        Stack n_ensemble model weights into a tensor tree with leading E, then
        define two vmapped forward functions:

          _vmap_forward_shared:
              Parameters/buffers vary over E while shared data is broadcast
              with in_dims=None. Used for inference and holdout evaluation on
              one common batch.

          _vmap_forward_per_member:
              Parameters, buffers, and data all vary over E, giving each
              member a distinct batch for genuine bootstrap diversity during
              training.

        `self._base_model` is only the template architecture for
        functional_call. Trainable weights live in self._stacked_params.
        self.models[k].parameters() is no longer authoritative after setup;
        no other code path reads it.
        """
        self._base_model = self.models[0]

        stacked_params, stacked_buffers = stack_module_state(self.models)

        self._stacked_params: Dict[str, torch.Tensor] = {
            k: v.detach().clone().requires_grad_(True)
            for k, v in stacked_params.items()
        }
        self._stacked_buffers: Dict[str, torch.Tensor] = dict(stacked_buffers)

        self.optim = torch.optim.Adam(
            list(self._stacked_params.values()), lr=self.lr
        )

        def _fmodel(params, buffers, obs_i, a_i_oh, z, m, belief, pair_feat):
            return functional_call(
                self._base_model,
                (params, buffers),
                args=(),
                kwargs=dict(
                    obs_i=obs_i,
                    action_i_onehot=a_i_oh,
                    z_core_excl_j=z,
                    m_periph_excl_j=m,
                    belief_summary=belief,
                    pair_feat=pair_feat,   # [FIX-X1]
                ),
            )

        # randomness="different" gives each member an independent dropout
        # mask if ensemble_dropout is enabled. vmap otherwise rejects random
        # operations without an explicit policy. [FIX-X1] also adds pair_feat
        # as the eighth _fmodel input dimension.
        self._vmap_forward_shared = vmap(
            _fmodel, in_dims=(0, 0, None, None, None, None, None, None),
            randomness="different",
        )
        self._vmap_forward_per_member = vmap(
            _fmodel, in_dims=(0, 0, 0, 0, 0, 0, 0, 0),
            randomness="different",
        )

        # ---------------------------------------------------------------
        # torch.compile is disabled by default and enabled explicitly with
        # compile_ensemble=True.
        #
        # Eager vmap does not fuse kernels: every Linear/ReLU still launches a
        # CUDA kernel with an extra batch dimension. For hidden=160, fixed
        # launch overhead can dominate compute and make GPU slower than CPU,
        # matching observed throughput of 12.5 on CUDA versus about 50 on Mac.
        # torch.compile(mode="reduce-overhead") uses CUDA graphs to fuse the
        # launch sequence and directly addresses this symptom.
        #
        # It is not the default because fixed batch shapes are required.
        # batch_size and holdout_size must remain constant, while the B*A batch
        # in _predict_all_actions changes with B and could trigger repeated
        # recompilation. The path also lacks verification on production GPU
        # hardware. Measure throughput before and after enabling it, starting
        # with fixed-batch _vmap_forward_per_member rather than variable-batch
        # _vmap_forward_shared.
        # ---------------------------------------------------------------
        if bool(getattr(self, "_compile_ensemble_flag", False)):
            self._vmap_forward_per_member = torch.compile(
                self._vmap_forward_per_member, mode="reduce-overhead"
            )

    def _ensemble_train_mode(self, training: bool):
        """Set template training mode, including Dropout, for all members."""
        if self.use_vmap_ensemble:
            self._base_model.train(training)
        else:
            for m in self.models:
                m.train(training)

    # =====================================================================
    # Helper tensor
    # =====================================================================

    def _one_hot(self, actions) -> torch.Tensor:
        """actions: array-like [B] -> [B, action_dim] float32"""
        if isinstance(actions, torch.Tensor):
            a = actions.to(device=self.device, dtype=torch.long)
        else:
            a = torch.tensor(
                np.asarray(actions, dtype=np.int64),
                dtype=torch.long,
                device=self.device,
            )

        if a.dim() == 0:
            a = a.unsqueeze(0)

        a = a.clamp(min=0, max=self.action_dim - 1)

        return F.one_hot(a, num_classes=self.action_dim).to(dtype=torch.float32)

    def _to_float_tensor(self, x, expected_dim=None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x.to(device=self.device, dtype=torch.float32)
        else:
            t = torch.tensor(
                np.asarray(x, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )

        if t.dim() == 1:
            t = t.unsqueeze(0)

        if expected_dim is not None and t.shape[-1] != int(expected_dim):
            raise ValueError(
                f"Expected last dim={expected_dim}, got {t.shape[-1]}"
            )

        return t

    def _normalise_vector(self, x, expected_dim) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)

        if arr.shape[0] != int(expected_dim):
            raise ValueError(
                f"Expected vector dim={expected_dim}, got {arr.shape[0]}"
            )

        return arr.astype(np.float32)

    # =====================================================================
    # Buffer
    # =====================================================================

    def add_sample(
        self,
        ego_id,
        neighbor_id,
        obs_i,
        action_i,
        observed_action_j,
        z_core_excl_j,
        m_periph_excl_j,
        belief_summary,
        target_return_h,
        pair_feat=None,
        target_returns_multi=None,
        behaviour_prob_j=None,
        was_forced=False,
        state_key=None,
    ):
        """
        Add one supervised sample.

        Args:
            target_returns_multi:
                list/array length n_horizons = [R^(1), R^(2), ..., R^(H)].
                If None, broadcast target_return_h across horizons. This is
                less accurate and retained only for compatibility; callers
                should provide the vector.
            behaviour_prob_j:
                b_j(a_j_obs | s) at collection time, required for DR. None
                disables DR for this sample and falls back to plug-in.
            was_forced:
                True when j's action was epsilon-forced, representing a true,
                higher-value intervention that is oversampled in training.
            state_key:
                Context identifier such as zone ID or coarse position hash.
        """
        if target_returns_multi is None:
            multi = np.full(
                (self.n_horizons,), float(target_return_h), dtype=np.float32
            )
        else:
            multi = np.asarray(target_returns_multi, dtype=np.float32).reshape(-1)

            if multi.shape[0] != self.n_horizons:
                raise ValueError(
                    f"target_returns_multi phải có length {self.n_horizons}, "
                    f"nhận {multi.shape[0]}"
                )

        sample = {
            "ego_id": int(ego_id),
            "neighbor_id": int(neighbor_id),
            "obs_i": self._normalise_vector(obs_i, self.obs_dim),
            "action_i": int(action_i),
            "observed_action_j": int(observed_action_j),
            "z_core_excl_j": self._normalise_vector(z_core_excl_j, self.core_dim),
            "m_periph_excl_j": self._normalise_vector(
                m_periph_excl_j, self.periph_dim
            ),
            "belief_summary": self._normalise_vector(
                belief_summary, self.belief_dim
            ),
            # [FIX-X1] x_ij; use zeros only for legacy pair_feat_dim == 0.
            # Otherwise forward raises if the caller omitted the feature.
            "pair_feat": (
                np.zeros((self.pair_feat_dim,), dtype=np.float32)
                if pair_feat is None
                else self._normalise_vector(pair_feat, self.pair_feat_dim)
            ),
            "target_return_h": float(target_return_h),
            "target_returns_multi": multi,                      # [n_horizons]
            "behaviour_prob_j": (
                None if behaviour_prob_j is None else float(behaviour_prob_j)
            ),
            "was_forced": bool(was_forced),
            "state_key": state_key,
        }

        self.buffer.append(sample)

        if bool(was_forced):
            self.n_interventional_samples += 1

    def add_sample_batch(self, samples: List[dict]):
        """
        Add multiple samples at once to reduce call overhead when the runner
        pushes a full trajectory with O(n_agents^2) samples per timestep.
        Each entry must contain the keys accepted by add_sample.

        This does not convert the buffer to tensors; it remains deque[dict]
        because drift_probe.py depends on that format. The optimization only
        replaces n_agents^2 separate Python add_sample calls per step with one
        inexpensive append loop.
        """
        for s in samples:
            self.add_sample(**s)

    def get_buffer_size(self) -> int:
        return int(len(self.buffer))

    def get_last_train_called(self) -> bool:
        return bool(self.last_train_called)

    def get_last_train_batch_count(self) -> int:
        return int(self.last_train_batch_count)

    def get_latest_residual(self) -> float:
        return float(self.latest_residual)

    def get_latest_train_residual(self) -> float:
        return float(self.latest_train_residual)

    def get_latest_holdout_residual(self) -> float:
        return float(self.latest_holdout_residual)

    # =====================================================================
    # Train
    # =====================================================================

    def _batch_to_tensors(self, batch):
        """batch: list[dict] length B -> tuple tensors"""
        obs = np.stack([b["obs_i"] for b in batch], axis=0)          # [B, obs_dim]
        action_i = np.asarray([b["action_i"] for b in batch], np.int64)      # [B]
        action_j = np.asarray(
            [b["observed_action_j"] for b in batch], np.int64
        )                                                                     # [B]
        z = np.stack([b["z_core_excl_j"] for b in batch], axis=0)    # [B, core_dim]
        m = np.stack([b["m_periph_excl_j"] for b in batch], axis=0)  # [B, periph_dim]
        belief = np.stack(
            [b["belief_summary"] for b in batch], axis=0
        )                                                            # [B, belief_dim]
        pair_feat = np.stack(
            [b["pair_feat"] for b in batch], axis=0
        )                                                       # [B, pair_feat_dim]
        # [FIX-HC1] b_j(a_j|s) for inverse-propensity loss weighting.
        b_obs = np.asarray(
            [(1.0 if b.get("behaviour_prob_j") is None
              else float(b["behaviour_prob_j"])) for b in batch],
            dtype=np.float32,
        )                                                                # [B]
        target_multi = np.stack(
            [b["target_returns_multi"] for b in batch], axis=0
        )                                                            # [B, n_horizons]

        return (
            torch.tensor(obs, dtype=torch.float32, device=self.device),
            self._one_hot(action_i),
            # a_j is now a raw int64 [B] index used for gather, not forward input.
            torch.tensor(action_j, dtype=torch.int64, device=self.device),
            torch.tensor(z, dtype=torch.float32, device=self.device),
            torch.tensor(m, dtype=torch.float32, device=self.device),
            torch.tensor(belief, dtype=torch.float32, device=self.device),
            torch.tensor(target_multi, dtype=torch.float32, device=self.device),
            torch.tensor(pair_feat, dtype=torch.float32, device=self.device),
            torch.tensor(b_obs, dtype=torch.float32, device=self.device),
        )

    def _sample_for_member(self, buf_list: list, member_idx: int, n: int,
                            forced_boost: float = None):
        # [FIX-HC3b] Reduce 8.0 to 4.0 through self.forced_boost. The previous
        # value amplified the exact sample group for which VERIFY-F1 still
        # suspected bad labels: min_head_frac=0.001 was 15-30x below the
        # theoretical forced_frac/|A| lower bound. Increase only after
        # VERIFY-F1 and F1b pass.
        """
        [L3] Sample independently for each ensemble member without a Python
        full-buffer scan. The old per-element hashing cost O(buffer_size) per
        member and O(n_ensemble * buffer_size) per train_step: 800k pure Python
        operations for a 200k buffer and four members, dominating wall time
        independently of GPU work.

        [BB1] Each member pool comes from the fixed permutation in
        self._member_pool_mask[member_idx], computed once at initialization.
        Redrawing it per call eventually exposes almost the full buffer to all
        members, removes systematic diversity, and collapses Eq. 10 sigma to
        zero as in v1. The paper's uncertainty-aware LCB, selectivity,
        targeted-epsilon, and inflation mechanisms all depend on avoiding that
        collapse. Fixed masks retain O(pool_size) NumPy cost while always
        excluding the same rank set for each member.

        `buf_list` is created once per train_step and shared across members
        because deque does not support O(1) random access.
        """
        if len(buf_list) == 0:
            return []

        rng = self._member_rngs[member_idx]

        mask = self._member_pool_mask[member_idx][: len(buf_list)]
        pool_positions = np.nonzero(mask)[0]

        if pool_positions.size == 0:
            # Only possible with a very small early-training buffer whose short
            # prefix misses the permutation; use the current full list and
            # never return an empty pool.
            pool_positions = np.arange(len(buf_list))

        pool = [buf_list[int(i)] for i in pool_positions]
        if forced_boost is None:
            forced_boost = float(getattr(self, "forced_boost", 4.0))
        weights = [forced_boost if s["was_forced"] else 1.0 for s in pool]

        # Weighted random.choices samples with replacement and always returns
        # n entries, even when the early-training pool is smaller. Equal batch
        # sizes then stack into [E,B,...]. Preserve BB2 oversampling for
        # was_forced=True; plain random.sample or slicing would remove it.
        return rng.choices(pool, weights=weights, k=int(n))

    def train_step(
        self,
        n_steps: int = 1,
        batch_size: int = 256,
        holdout_size: int = 0,
    ) -> float:
        """
        Train the ensemble while preserving the v1 signature.

        All n_ensemble member forward/backward/update operations run in one
        vmap call instead of a Python model loop. Diagnostics accumulate on
        GPU and synchronize to CPU once at the end rather than once per member
        per step.
        """
        self.last_train_called = True
        self.last_train_batch_count = 0

        if len(self.buffer) == 0:
            self.latest_loss = 0.0
            self.latest_residual = 0.0
            self.latest_train_residual = 0.0
            self.latest_holdout_residual = 0.0
            return 0.0

        n_steps = int(max(0, n_steps))
        batch_size = int(max(1, batch_size))
        holdout_size = int(max(0, holdout_size))

        if n_steps == 0:
            self.latest_loss = 0.0
            return 0.0

        if not self.use_vmap_ensemble:
            return self._train_step_fallback(n_steps, batch_size, holdout_size)

        self._ensemble_train_mode(True)

        E = self.n_ensemble
        per_step_losses = []       # [E] tensors; synchronize once at the end.
        per_step_residuals = []    # list of [E] tensors

        for _ in range(n_steps):
            buf_list = list(self.buffer)  # One deque-to-list copy per step.

            member_batches = [
                self._sample_for_member(buf_list, k, batch_size)
                for k in range(E)
            ]

#            print(f"[TRAIN-DEBUG] batch_size={batch_size} " f"member_batch_lens={[len(b) for b in member_batches]}")

            if any(len(b) == 0 for b in member_batches):
#                print("[TRAIN-DEBUG] SKIPPED — empty batch this step")
                continue

            if any(len(b) == 0 for b in member_batches):
                continue

            obs_l, ai_l, aj_l, z_l, m_l, bl_l, tgt_l = [], [], [], [], [], [], []
            pf_l = []   # [FIX-X1] x_ij per member
            bobs_l = []  # [FIX-HC1] b_j(a_j|s) per member

            for b in member_batches:
                (obs_t, a_i_oh, a_j_idx, z_t, m_t, belief_t, target_multi_t,
                 pf_t, bobs_t) = (
                    self._batch_to_tensors(b)
                )
                pf_l.append(pf_t); bobs_l.append(bobs_t)
                obs_l.append(obs_t)
                ai_l.append(a_i_oh)
                aj_l.append(a_j_idx)
                z_l.append(z_t)
                m_l.append(m_t)
                bl_l.append(belief_t)
                tgt_l.append(target_multi_t)

            obs_e = torch.stack(obs_l, dim=0)    # [E, B, obs_dim]
            ai_e = torch.stack(ai_l, dim=0)      # [E, B, A]
            aj_e = torch.stack(aj_l, dim=0)      # [E, B]
            z_e = torch.stack(z_l, dim=0)        # [E, B, core_dim]
            m_e = torch.stack(m_l, dim=0)        # [E, B, periph_dim]
            bel_e = torch.stack(bl_l, dim=0)     # [E, B, belief_dim]
            tgt_e = torch.stack(tgt_l, dim=0)    # [E, B, H]
            pf_e = torch.stack(pf_l, dim=0)      # [E, B, pair_feat_dim]
            bobs_e = torch.stack(bobs_l, dim=0)  # [E, B]

            # One vmap operation runs n_ensemble forwards in parallel on GPU.
            preds_all = self._vmap_forward_per_member(
                self._stacked_params, self._stacked_buffers,
                obs_e, ai_e, z_e, m_e, bel_e, pf_e,   # [FIX-X1]
            )  # [E, B, A, H]

            E_, B_, A_, H_ = preds_all.shape
            gather_idx = aj_e.view(E_, B_, 1, 1).expand(E_, B_, 1, H_)
            preds = torch.gather(preds_all, dim=2, index=gather_idx).squeeze(2)  # [E, B, H]

            # ----------------------------------------------------------------
            # [FIX-HC1] HEAD COLLAPSE: connect epsilon-forcing to training loss.
            #
            # Causal chain confirmed by the gather operation above:
            #   loss = MSE(gather(preds_all, a_j), target)
            #     -> each sample sends gradient to only one of |A| heads
            #     -> rare-action heads remain near initialization
            #     -> std_a f_theta(a) approximates initialization noise
            #     -> plug-in contrast f(a_j) - sum_a pi(a) f(a) ~ 0
            #     -> mu ~ 0 (measured mean_mu 0.117 vs W* ~1.5, a 13x gap)
            #
            # Epsilon-forcing is the only mechanism that evenly covers rare
            # heads, but forced samples were previously used only for DR
            # correction through b_j on the scoring path, not head training.
            # The generated counterfactuals were therefore left unused.
            #
            # Weight by clipped 1/b_j so rare-action heads receive gradients
            # inversely proportional to rarity. Normalize to mean one to
            # preserve the effective learning-rate scale.
            # ----------------------------------------------------------------
            iw = torch.clamp(
                1.0 / torch.clamp(bobs_e, min=1.0 / self.iw_clip, max=1.0),
                max=self.iw_clip,
            )                                                    # [E, B]
            iw = iw / torch.clamp(iw.mean(dim=1, keepdim=True), min=1e-8)

            sq = F.mse_loss(preds, tgt_e, reduction="none").mean(dim=2)  # [E,B]
            per_member_loss = (sq * iw).mean(dim=1)  # [E]

            loss = per_member_loss.sum()  # Backpropagate the component sum.
            # Components remain independent: each propagates only to its own
            # member because vmap never mixes the E dimension.

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            _clip_grad_norm_per_member(self._stacked_params, self.grad_clip)
            self.optim.step()

            per_step_losses.append(per_member_loss.detach())

            with torch.no_grad():
                res = torch.mean(
                    torch.abs(preds[:, :, -1] - tgt_e[:, :, -1]), dim=1
                )  # [E] residual at the final horizon, matching v1 R^(H).
                # [FIX-HC2] res_forced vs res_control lost diagnostic value
                # after the TARNet refactor. a_j is no longer input; forcing
                # only changes the gathered head, not R_i^(H) prediction
                # difficulty. Similar residuals are therefore inevitable, not
                # evidence of failure. head_spread directly measures whether
                # the network distinguishes the action axis.
                fm = torch.tensor(
                    [b["was_forced"] for b in member_batches[0]],
                    device=preds.device,
                )
                hs = preds_all.std(dim=2)                    # [E, B, H]
                hs_last = hs[:, :, -1].reshape(-1)
                hs_p50 = torch.quantile(hs_last, 0.50)
                hs_p90 = torch.quantile(hs_last, 0.90)
                mu_scale = torch.mean(torch.abs(preds[:, :, -1])) + 1e-8
                counts = torch.bincount(
                    aj_e.reshape(-1), minlength=int(self.action_dim)
                ).float()
                min_head_frac = float(counts.min().item()) / max(
                    1.0, float(aj_e.numel())
                )
                self.last_head_spread_p50 = float(hs_p50.item())
                self.last_head_spread_ratio = float((hs_p50 / mu_scale).item())
                self.last_min_head_frac = min_head_frac
                self.last_forced_frac = float(fm.float().mean().item())
                print(
                    f"[HEAD-SPREAD] p50={hs_p50.item():.4e} p90={hs_p90.item():.4e} "
                    f"p50/|mu|={self.last_head_spread_ratio:.3f} "
                    f"(gate >0.10) min_head_frac={min_head_frac:.3f} "
                    f"(gate >0.05) forced_frac={self.last_forced_frac:.3f}"
                )
            per_step_residuals.append(res)

            self.last_train_batch_count += 1

        # Holdout residual: shared batch excluded from updates.
        # [H7] Residual must use data outside the gradient update; otherwise it
        # reflects its own parameter change rather than generalization.
        # structural shift.
        holdout_residual_t = None

        if holdout_size > 0 and len(self.buffer) > holdout_size:
            ho_batch = random.sample(list(self.buffer), int(holdout_size))

            (ho_obs, ho_ai, ho_aj, ho_z, ho_m, ho_b, ho_target, ho_pf,
             _ho_bobs) = (
                self._batch_to_tensors(ho_batch)
            )

            self._ensemble_train_mode(False)

            with torch.no_grad():
                stacked_all = self._vmap_forward_shared(
                    self._stacked_params, self._stacked_buffers,
                    ho_obs, ho_ai, ho_z, ho_m, ho_b, ho_pf,
                )  # [E,B,A,H] from one vmap over data shared by all members.

                E_, B_, A_, H_ = stacked_all.shape
                ho_idx = ho_aj.view(1, B_, 1, 1).expand(E_, B_, 1, H_)
                stacked = torch.gather(stacked_all, dim=2, index=ho_idx).squeeze(2)  # [E, B, H]
                pred_mean = stacked.mean(dim=0)  # [B, H]

                holdout_residual_t = torch.mean(
                    torch.abs(pred_mean[:, -1] - ho_target[:, -1])
                )

                # [L3] Diagnose genuine ensemble disagreement. A value near
                # zero indicates a collapsed ensemble and meaningless sigma.
                if stacked.shape[0] > 1:
                    self.latest_ensemble_disagreement = float(
                        torch.mean(torch.std(stacked, dim=0)).item()
                    )  # One synchronization for the complete holdout evaluation.

        if len(per_step_losses) == 0:
            print("[TRAIN-DEBUG] ALL n_steps SKIPPED — per_step_losses rỗng")
            self.latest_loss = 0.0
            self.latest_residual = 0.0
            self.latest_train_residual = 0.0
            self.latest_holdout_residual = 0.0
            return 0.0

        # One CPU/GPU synchronization for the full train_step call. The old
        # path used n_steps*n_ensemble .item() calls for loss and as many for
        # residual. This path uses one for each regardless of dimensions.
        losses_stacked = torch.stack(per_step_losses)  # [actual_n_steps, E]
        self.latest_loss = float(losses_stacked.mean().item())
        # [BB3] Per-member loss averaged across steps lets T3 confirm that
        # member 0's gradient scale does not change when members are added.
        self.latest_loss_per_member = (
            losses_stacked.mean(dim=0).detach().cpu().numpy()
        )
        self.latest_train_residual = float(
            torch.stack(per_step_residuals).mean().item()
        )

        if holdout_residual_t is not None:
            self.latest_holdout_residual = float(holdout_residual_t.item())
            self.latest_residual = self.latest_holdout_residual
        else:
            self.latest_holdout_residual = self.latest_train_residual
            self.latest_residual = self.latest_train_residual

        return float(self.latest_loss)

    # Fallback for torch <2.0 without torch.func.
    def _train_step_fallback(self, n_steps, batch_size, holdout_size):
        """Legacy Python loop for torch <2.0; correct but not GPU-optimized."""
        all_losses = []
        train_residuals = []

        for _ in range(n_steps):
            buf_list = list(self.buffer)
            for k, (model, optim) in enumerate(zip(self.models, self.optims)):
                batch = self._sample_for_member(buf_list, k, batch_size)

                if len(batch) == 0:
                    continue

                (obs_t, a_i_oh, a_j_idx, z_t, m_t, belief_t, target_multi_t,
                 pf_t, _bobs_t) = (
                    self._batch_to_tensors(batch)
                )

                model.train()

                pred_all = model(
                    obs_i=obs_t,
                    action_i_onehot=a_i_oh,
                    z_core_excl_j=z_t,
                    m_periph_excl_j=m_t,
                    belief_summary=belief_t,
                    pair_feat=pf_t,
                )

                B_, A_, H_ = pred_all.shape
                idx = a_j_idx.view(B_, 1, 1).expand(B_, 1, H_)
                pred = torch.gather(pred_all, dim=1, index=idx).squeeze(1)  # [B, H]

                loss = F.mse_loss(pred, target_multi_t)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optim.step()

                all_losses.append(loss.detach())

                with torch.no_grad():
                    res = torch.mean(torch.abs(pred[:, -1] - target_multi_t[:, -1]))
                train_residuals.append(res)

            self.last_train_batch_count += 1

        holdout_residual = None

        if holdout_size > 0 and len(self.buffer) > holdout_size:
            ho_batch = random.sample(list(self.buffer), int(holdout_size))
            (ho_obs, ho_ai, ho_aj, ho_z, ho_m, ho_b, ho_target, ho_pf,
             _ho_bobs2) = (
                self._batch_to_tensors(ho_batch)
            )

            with torch.no_grad():
                preds = []
                for model in self.models:
                    model.eval()
                    pred_all = model(
                        obs_i=ho_obs, action_i_onehot=ho_ai,
                        z_core_excl_j=ho_z, m_periph_excl_j=ho_m,
                        belief_summary=ho_b, pair_feat=ho_pf,
                    )
                    B_, A_, H_ = pred_all.shape
                    idx = ho_aj.view(B_, 1, 1).expand(B_, 1, H_)
                    preds.append(torch.gather(pred_all, dim=1, index=idx).squeeze(1))

                stacked = torch.stack(preds, dim=0)
                pred_mean = stacked.mean(dim=0)
                holdout_residual = float(
                    torch.mean(torch.abs(pred_mean[:, -1] - ho_target[:, -1]))
                )


                if stacked.shape[0] > 1:
                    self.latest_ensemble_disagreement = float(
                        torch.mean(torch.std(stacked, dim=0)).item()
                    )

        if len(all_losses) == 0:
            self.latest_loss = 0.0
            self.latest_residual = 0.0
            return 0.0

        self.latest_loss = float(torch.stack(all_losses).mean().item())
        self.latest_train_residual = (
            float(torch.stack(train_residuals).mean().item())
            if train_residuals else 0.0
        )

        if holdout_residual is not None:
            self.latest_holdout_residual = holdout_residual
            self.latest_residual = holdout_residual
        else:
            self.latest_holdout_residual = self.latest_train_residual
            self.latest_residual = self.latest_train_residual

        return float(self.latest_loss)

    # =====================================================================
    # Predict every alternative action.
    # =====================================================================

    def _predict_all_actions(
        self,
        obs,       # [B, obs_dim]
        action_i,  # [B]
        z,         # [B, core_dim]
        m,         # [B, periph_dim]
        belief,    # [B, belief_dim]
        pair_feat=None,   # [B, pair_feat_dim] — x_ij (FIX-X1)
    ) -> torch.Tensor:
        """
        Predict returns for every possible action of j and every ensemble
        member in exactly one GPU forward pass.

        Two batching levels:
          1. combine all alternative actions into one batch rather than
             calling forward action_dim times;
          2. combine the ensemble dimension with vmap rather than looping over
             models.

        Returns:
            [E, B, A, n_horizons]
        """
        obs_t = self._to_float_tensor(obs, self.obs_dim)        # [B, obs_dim]
        z_t = self._to_float_tensor(z, self.core_dim)           # [B, core_dim]
        m_t = self._to_float_tensor(m, self.periph_dim)         # [B, periph_dim]
        belief_t = self._to_float_tensor(belief, self.belief_dim)  # [B, belief_dim]

        B = int(obs_t.shape[0])
        A = int(self.action_dim)

        a_i_oh = self._one_hot(np.asarray(action_i).reshape(-1))  # [B, A]
        pf_t = self._pair_feat_tensor(pair_feat, B)   # [FIX-X1]

        self._ensemble_train_mode(False)

        with torch.no_grad():
            if self.use_vmap_ensemble:
                out = self._vmap_forward_shared(
                    self._stacked_params, self._stacked_buffers,
                    obs_t, a_i_oh, z_t, m_t, belief_t, pf_t,
                )  # [E, B, A, n_horizons]
            else:
                out = self._predict_all_actions_loop(
                    obs_t, a_i_oh, z_t, m_t, belief_t, pf_t)

        return out  # [E, B, A, n_horizons]

    def _predict_all_actions_loop(
        self, obs, a_i_oh, z, m, belief, pair_feat=None,
    ) -> torch.Tensor:
        """Per-model Python loop used for torch <2.0 fallback and as the T1
        reference. It is excluded from the hot path when vmap is active."""
        outs = []
        for model in self.models:
            model.eval()
            pred = model(
                obs_i=obs, action_i_onehot=a_i_oh,
                z_core_excl_j=z, m_periph_excl_j=m,
                belief_summary=belief, pair_feat=pair_feat,
            )
            outs.append(pred)
        return torch.stack(outs, dim=0)

    def _sync_stacked_to_models(self):
        """Copy authoritative stacked parameters back into self.models[k] for
        tests/debug reference only, outside production train/inference."""
        if not self.use_vmap_ensemble:
            return
        with torch.no_grad():
            for k, model in enumerate(self.models):
                sd = model.state_dict()
                for name, p in self._stacked_params.items():
                    if name in sd:
                        sd[name].copy_(p[k])

    def _predict_all_actions_reference(
        self, obs, action_i, z, m, belief, pair_feat=None,
    ) -> torch.Tensor:
        """
        Slow reference required by GPU_OPTIMIZATION_CONTRACT.md section 3. It
        synchronizes current stacked parameters and loops over models without
        vmap. Smoke tests compare it with the fast path using allclose. This is
        the only reliable way to expose BB4 layout errors from inverted
        repeat_interleave/repeat/view order, because both paths independently
        produce plausible numeric values. This function is test-only.
        """
        self._sync_stacked_to_models()

        obs_t = self._to_float_tensor(obs, self.obs_dim)
        z_t = self._to_float_tensor(z, self.core_dim)
        m_t = self._to_float_tensor(m, self.periph_dim)
        belief_t = self._to_float_tensor(belief, self.belief_dim)

        a_i_oh = self._one_hot(np.asarray(action_i).reshape(-1))
        pf_t = self._pair_feat_tensor(pair_feat, int(obs_t.shape[0]))

        with torch.no_grad():
            out = self._predict_all_actions_loop(
                obs_t, a_i_oh, z_t, m_t, belief_t, pf_t)

        return out  # [E, B, A, n_horizons]

    def _pair_feat_tensor(self, pair_feat, B):
        """[FIX-X1] Normalize x_ij to [B, pair_feat_dim].

        Return None when pair_feat_dim == 0 for legacy compatibility. When it
        is positive, raise if the caller omits x_ij. Silent zeros mean every
        neighbour is identical and would reintroduce FIX-X1's defect.
        """
        if self.pair_feat_dim <= 0:
            return None
        if pair_feat is None:
            raise ValueError(
                f"pair_feat_dim={self.pair_feat_dim} nhưng call site không "
                "truyền x_ij. Xem FIX-X1."
            )
        arr = np.asarray(pair_feat, dtype=np.float32).reshape(B, -1)
        if arr.shape[1] != self.pair_feat_dim:
            raise ValueError(
                f"x_ij phải có {self.pair_feat_dim} chiều, nhận {arr.shape[1]}"
            )
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    # =====================================================================
    # Effect computation.
    # =====================================================================

    def _compute_effects(
        self,
        preds_all: torch.Tensor,        # [E, B, A, n_horizons]
        action_j_obs: np.ndarray,       # [B]
        policy_probs_j: Optional[np.ndarray] = None,   # [B, A]
        observed_returns: Optional[np.ndarray] = None,  # [B]
        behaviour_probs_obs: Optional[np.ndarray] = None,  # [B]
        mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the selected effect with or without DR. The implementation is
        already vectorized with einsum/gather.

        Returns dict:
            effect:        [E,B] final-horizon effect
            effect_per_h:  [E,B,n_horizons] horizon-specific effects
            dr_correction: [B] DR correction magnitude
        """
        mode = self.effect_mode if mode is None else str(mode)

        E, B, A, H = preds_all.shape

        idx_obs = torch.tensor(
            np.asarray(action_j_obs, dtype=np.int64).reshape(-1),
            dtype=torch.long,
            device=self.device,
        ).clamp(0, A - 1)  # [B]

        idx_exp = (
            idx_obs.view(1, B, 1, 1).expand(E, B, 1, H)
        )  # [E, B, 1, H]

        f_obs = torch.gather(preds_all, dim=2, index=idx_exp).squeeze(2)
        # f_obs: [E, B, H]

        dr_correction = torch.zeros(B, dtype=torch.float32, device=self.device)

        # ---------------------------------------------------------------
        if mode == "signed_aristocrat":
            if policy_probs_j is None:
                w = torch.full(
                    (B, A), 1.0 / float(A), dtype=torch.float32, device=self.device
                )
            else:
                w = torch.tensor(
                    np.asarray(policy_probs_j, dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )  # [B, A]

            baseline = torch.einsum("ebah,ba->ebh", preds_all, w)
            effect_per_h = f_obs - baseline  # [E, B, H]

        elif mode == "signed_oracle_matched":
            cand = torch.tensor(
                [a for a in self.candidate_actions if 0 <= a < A],
                dtype=torch.long,
                device=self.device,
            )  # [n_cand]

            if cand.numel() == 0:
                cand = torch.arange(A, dtype=torch.long, device=self.device)

            cand_preds = preds_all[:, :, cand, :]      # [E, B, n_cand, H]
            cand_mean = cand_preds.mean(dim=2)         # [E, B, H]

            effect_per_h = cand_mean - f_obs           # [E, B, H]

        elif mode == "range":
            max_f = preds_all.max(dim=2).values        # [E, B, H]
            min_f = preds_all.min(dim=2).values        # [E, B, H]
            effect_per_h = max_f - min_f               # [E, B, H]

        elif mode == "mean_abs":
            diff = torch.abs(preds_all - f_obs.unsqueeze(2))  # [E, B, A, H]

            mask = torch.ones(B, A, dtype=torch.float32, device=self.device)
            mask.scatter_(1, idx_obs.view(B, 1), 0.0)          # [B, A]

            denom = torch.clamp(mask.sum(dim=1), min=1.0)      # [B]

            effect_per_h = (
                torch.einsum("ebah,ba->ebh", diff, mask) / denom.view(1, B, 1)
            )  # [E, B, H]

        else:
            raise ValueError(f"mode không hợp lệ: {mode}")

        # ---------------------------------------------------------------
        # Doubly robust correction applies only to signed modes.
        # ---------------------------------------------------------------
        apply_dr = (
            self.use_doubly_robust
            and mode in ("signed_aristocrat", "signed_oracle_matched")
            and observed_returns is not None
            and behaviour_probs_obs is not None
        )

        if apply_dr:
            R_obs = torch.tensor(
                np.asarray(observed_returns, dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )  # [B]

            b_obs = torch.tensor(
                np.asarray(behaviour_probs_obs, dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            ).clamp(min=1.0 / self.iw_clip, max=1.0)  # [B]

            residual = R_obs.view(1, B) - f_obs[:, :, -1]  # [E, B]

            iw_minus_one = torch.clamp(
                1.0 / b_obs - 1.0, min=0.0, max=self.iw_clip
            )  # [B]

            correction = residual * iw_minus_one.view(1, B)  # [E, B]

            if mode == "signed_oracle_matched":
                correction = -correction

            # [FIX-DR-H] Eq. 10 defines DR only for R^(H), so it correctly
            # applies only to the final horizon. This makes effect_per_h a
            # mixed-estimator vector: plug-in for h<H and DR-corrected at H.
            # Eq. 19 latency previously combined |ŵ^(h)| across all h, mixing
            # estimands and making Eq. 18's sixth component inconsistent.
            # Retain a pure plug-in vector for latency diagnostics while using
            # corrected mu/effect for the actual Eq. 10 quantity.
            effect_per_h_plugin = effect_per_h
            effect_per_h = effect_per_h.clone()
            effect_per_h[:, :, -1] = effect_per_h[:, :, -1] + correction

            dr_correction = torch.mean(torch.abs(correction), dim=0)  # [B]

        return {
            "effect": effect_per_h[:, :, -1],   # [E, B]
            "effect_per_h": effect_per_h,       # [E,B,H], with DR at h=H
            # [FIX-DR-H] Consistent estimator across horizons for latency diagnostics.
            "effect_per_h_plugin": (
                effect_per_h_plugin if apply_dr else effect_per_h
            ),
            "dr_correction": dr_correction,     # [B]
        }

    # =====================================================================
    # API scoring
    # =====================================================================

    def score_batch(
        self,
        obs_i_batch,
        action_i_batch,
        observed_action_j_batch,
        z_core_excl_j_batch,
        m_periph_excl_j_batch,
        belief_summary_batch,
        policy_probs_j_batch=None,
        observed_returns_batch=None,
        behaviour_probs_obs_batch=None,
        pair_feat_batch=None,
    ):
        """
        Preserve the v1 signature for immediate legacy-runner compatibility.

        Returns:
            mu_arr: np.ndarray [B], signed for signed_* effect modes
            sigma_arr: np.ndarray [B], ensemble standard deviation/epistemic uncertainty
        """
        out = self.score_batch_full(
            obs_i_batch=obs_i_batch,
            action_i_batch=action_i_batch,
            observed_action_j_batch=observed_action_j_batch,
            z_core_excl_j_batch=z_core_excl_j_batch,
            m_periph_excl_j_batch=m_periph_excl_j_batch,
            belief_summary_batch=belief_summary_batch,
            policy_probs_j_batch=policy_probs_j_batch,
            observed_returns_batch=observed_returns_batch,
            behaviour_probs_obs_batch=behaviour_probs_obs_batch,
            pair_feat_batch=pair_feat_batch,
        )

        return out["mu"], out["sigma"]

    def score_batch_full(
        self,
        obs_i_batch,
        action_i_batch,
        observed_action_j_batch,
        z_core_excl_j_batch,
        m_periph_excl_j_batch,
        belief_summary_batch,
        policy_probs_j_batch=None,
        observed_returns_batch=None,
        behaviour_probs_obs_batch=None,
        pair_feat_batch=None,
    ) -> Dict[str, np.ndarray]:
        """
        Full version providing every field needed by influence_signature.py.

        Returns dict of np.ndarray:
            mu            [B] ensemble-mean signed effect
            sigma         [B] ensemble std/epistemic uncertainty
            mu_per_h      [B,H] raw per-horizon effect diagnostic
            mu_range      [B] nonnegative Pieroth-style baseline impact
            dr_correction [B] DR magnitude diagnostic for model bias
        """
        B = len(obs_i_batch)

        if B == 0:
            z = np.zeros((0,), dtype=np.float32)
            return {
                "mu": z,
                "sigma": z,
                "mu_per_h": np.zeros((0, self.n_horizons), dtype=np.float32),
                "mu_range": z,
                "dr_correction": z,
            }

        obs = np.asarray(obs_i_batch, dtype=np.float32)
        z_arr = np.asarray(z_core_excl_j_batch, dtype=np.float32)
        m_arr = np.asarray(m_periph_excl_j_batch, dtype=np.float32)
        belief = np.asarray(belief_summary_batch, dtype=np.float32)
        a_i = np.asarray(action_i_batch, dtype=np.int64).reshape(-1)
        a_j = np.asarray(observed_action_j_batch, dtype=np.int64).reshape(-1)

        # One forward pass over both alternative actions and ensemble members.
        preds_all = self._predict_all_actions(
            obs=obs, action_i=a_i, z=z_arr, m=m_arr, belief=belief,
            pair_feat=pair_feat_batch,   # [FIX-X1]
        )  # [E, B, A, H]

        res = self._compute_effects(
            preds_all=preds_all,
            action_j_obs=a_j,
            policy_probs_j=policy_probs_j_batch,
            observed_returns=observed_returns_batch,
            behaviour_probs_obs=behaviour_probs_obs_batch,
            mode=self.effect_mode,
        )

        effect = res["effect"]              # [E, B]
        effect_per_h = res["effect_per_h"]  # [E, B, H]

        mu = torch.mean(effect, dim=0)      # [B]

        if effect.shape[0] <= 1:
            sigma = torch.zeros_like(mu)
        else:
            sigma = torch.sqrt(
                torch.var(effect, dim=0, unbiased=True) + self.eps
            )  # [B]

        mu_per_h = torch.mean(effect_per_h, dim=0)  # [B, H]

        # [SIG-5D] The latency component was removed, reducing the signature to
        # R^5; see influence_signature.py. mu_per_h remains only as a raw
        # horizon diagnostic and carries no latency-centroid claim.

        # Always compute mu_range for the Pieroth baseline.
        res_range = self._compute_effects(
            preds_all=preds_all,
            action_j_obs=a_j,
            mode="range",
        )
        mu_range = torch.mean(res_range["effect"], dim=0)        # [B]

        self.latest_dr_correction_magnitude = float(
            torch.mean(res["dr_correction"]).item()
        )

        # One .cpu().numpy() per output at the API boundary. Leaving GPU is
        # required here because downstream influence_signature.py and
        # belief_layer.py still accept NumPy and were not vectorized in this
        # revision; see README_INTEGRATION.md.
        to_np = lambda t: t.detach().cpu().numpy().astype(np.float32)

        return {
            "mu": to_np(mu),
            "sigma": to_np(sigma),
            "mu_per_h": to_np(mu_per_h),
            "mu_range": to_np(mu_range),
            "dr_correction": to_np(res["dr_correction"]),
        }

    def score_pair(
        self,
        obs_i,
        action_i,
        observed_action_j,
        z_core_excl_j,
        m_periph_excl_j,
        belief_summary,
        **kwargs,
    ):
        """Single-pair wrapper preserving the v1 signature."""
        mu_arr, sigma_arr = self.score_batch(
            obs_i_batch=[obs_i],
            action_i_batch=[int(action_i)],
            observed_action_j_batch=[int(observed_action_j)],
            z_core_excl_j_batch=[z_core_excl_j],
            m_periph_excl_j_batch=[m_periph_excl_j],
            belief_summary_batch=[belief_summary],
            **kwargs,
        )

        if len(mu_arr) == 0:
            return 0.0, 0.0

        return float(mu_arr[0]), float(sigma_arr[0])

    # =====================================================================
    # Diagnostics.
    # =====================================================================

    def get_diagnostics(self) -> Dict[str, float]:
        """
        Diagnostic statistics. The three most important values are:

        ensemble_disagreement:
            Near zero indicates v1-style ensemble collapse and meaningless
            sigma. After correction it should be positive and decline as
            learning improves.

        dr_correction_magnitude:
            Measures reward-model bias. Large values mean DR carries a poor
            model; small values indicate a good model.

        interventional_fraction:
            Fraction of samples from epsilon-forcing, representing true interventions.
        """
        n = max(1, len(self.buffer))

        return {
            "buffer_size": int(len(self.buffer)),
            "n_interventional_samples": int(self.n_interventional_samples),
            "interventional_fraction": float(self.n_interventional_samples) / float(n),
            "latest_loss": float(self.latest_loss),
            "latest_train_residual": float(self.latest_train_residual),
            "latest_holdout_residual": float(self.latest_holdout_residual),
            "ensemble_disagreement": float(self.latest_ensemble_disagreement),
            "dr_correction_magnitude": float(self.latest_dr_correction_magnitude),
            "effect_mode": self.effect_mode,
            "use_doubly_robust": bool(self.use_doubly_robust),
            "n_ensemble": int(self.n_ensemble),
            "n_horizons": int(self.n_horizons),
            "use_vmap_ensemble": bool(self.use_vmap_ensemble),
        }


# =========================================================================
# [FIX-X1] x_ij — pair features for Equation 8
# =========================================================================

PAIR_FEAT_DIM = 5


def build_pair_feat(positions, agent_zone, grid_size, n_zones, ego, j):
    """x_ij = [drow/G, dcol/G, L1dist/G, same_zone, zone_diff/(n_zones-1)].

    WHY THIS IS REQUIRED (see FIX-X1 in LocalCounterfactualProxyNet.__init__):
    In omni_arena, w_ij(s)=phi_ij*delta_ij(s), where delta_ij(s) is purely a
    function of position: every _gate_ladder branch uses _dist(pos, anchor).
    Without x_ij, f_theta has no input distinguishing j and therefore cannot
    represent w_ij(s), regardless of training duration.

    Use this same function on both the replay_builder push path and
    final_runner scoring path. Divergent x_ij definitions cause silent,
    difficult-to-detect train/serve skew.
    """
    import numpy as _np

    g = float(max(1, grid_size))
    pi = positions[int(ego)]
    pj = positions[int(j)]
    drow = (float(pj[0]) - float(pi[0])) / g
    dcol = (float(pj[1]) - float(pi[1])) / g
    dist = (abs(float(pj[0]) - float(pi[0])) + abs(float(pj[1]) - float(pi[1]))) / g
    zi = int(agent_zone[int(ego)])
    zj = int(agent_zone[int(j)])
    same_zone = 1.0 if zi == zj else 0.0
    zone_diff = (zj - zi) / float(max(1, int(n_zones) - 1))

    return _np.asarray(
        [drow, dcol, dist, same_zone, zone_diff], dtype=_np.float32
    )
