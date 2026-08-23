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
     exposes an augmented inverse-propensity pseudo-outcome using the logged
     behaviour propensity. A formal conditional double-robust estimator still
     requires cross-fitting and a second-stage effect regression; the online
     row-level correction is retained as an explicitly diagnosed ablation.

[L3] THE ENSEMBLE WAS EFFECTIVELY IDENTICAL.
     All three v1 members trained on the same batches in the same order and
     converged to nearly the same function, producing sigma = 0.000. Current
     members have independent bootstrap masks, batches, and initializations.
     Their complete forward/backward/update runs as one torch.func.vmap GPU
     operation without a Python loop over models.

[L4] A SINGLE HORIZON PROVIDED NO LATENCY DIMENSION.
     A per-lag head predicts g^[0], ..., g^[H-1] jointly. Cumulative Q^(h)
     values are derived with the configured discount rather than learned as
     mutually unconstrained cumulative heads.

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

from envs.causal_adapter import OmniArenaAdapter, PUBLIC_ROLES

try:
    from torch.func import functional_call, stack_module_state, vmap
    _HAS_TORCH_FUNC = True
except ImportError:  # Rare torch <2.0 fallback; remain operational.
    _HAS_TORCH_FUNC = False


def _splitmix64(value: int) -> int:
    """Stable 64-bit mixer for independent immutable bootstrap membership."""
    mask = (1 << 64) - 1
    value = (int(value) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


# =============================================================================
# Proxy network
# =============================================================================

class LocalCounterfactualProxyNet(nn.Module):
    """
    One ensemble member.

    Conditioning input:
        obs_i, a_i, x_ij, sum_{k != i,j} phi_ctx(x_ik)

    Output:
        [B, n_horizons] predictions of direct rewards at lags 0, ..., H-1

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
        discount: float = 0.97,
        use_belief_input: bool = False,
        dropout: float = 0.0,
        pair_feat_dim: int = 0,
        context_item_dim: int = 0,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.core_dim = int(core_dim)
        self.periph_dim = int(periph_dim)
        self.belief_dim = int(belief_dim)
        self.hidden = int(hidden)
        self.n_horizons = int(n_horizons)
        self.discount = float(discount)

        # ---------------------------------------------------------------
        # [H6] Break the belief -> proxy -> belief feedback loop.
        #
        # B_i derives from this proxy's own w_hat. Feeding B_i back lets the
        # proxy self-confirm and creates an architectural confounder. Belief
        # input is disabled by default.
        # ---------------------------------------------------------------
        self.use_belief_input = bool(use_belief_input)
        self.context_input_dim = self.core_dim + self.periph_dim
        self.context_embed_dim = max(16, self.hidden // 2)
        # This summary encoder is retained only for explicit legacy calls that
        # omit raw context items. Confirmatory runners use context_item_encoder:
        # each item passes through phi before the leave-one-out sum.
        self.context_set_encoder = nn.Sequential(
            nn.Linear(self.context_input_dim, self.context_embed_dim),
            nn.ReLU(),
            nn.Linear(self.context_embed_dim, self.context_embed_dim),
            nn.ReLU(),
        )

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
        self.context_item_dim = int(context_item_dim)
        self.context_item_encoder = (
            nn.Sequential(
                nn.Linear(self.context_item_dim, self.context_embed_dim),
                nn.ReLU(),
                nn.Linear(self.context_embed_dim, self.context_embed_dim),
                nn.ReLU(),
            )
            if self.context_item_dim > 0 else None
        )

        self.in_dim = (
            self.obs_dim
            + self.action_dim   # a_i one-hot
            # a_j is no longer an input; see the multi-head output below.
            + self.pair_feat_dim   # x_ij (Eq 8)
            + self.context_embed_dim
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
        context_items: torch.Tensor = None, # [..., B, K, context_item_dim]
        context_mask: torch.Tensor = None,  # [..., B, K]
        context_embedding: torch.Tensor = None,
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
                    "StructuralProxyNet was initialized with pair_feat_dim="
                    f"{self.pair_feat_dim}, but forward() received no pair_feat. "
                    "Without x_ij, f_theta cannot distinguish neighbour j; "
                    "silently omitting it is not allowed."
                )
            parts.append(pair_feat)

        if context_embedding is not None:
            if context_embedding.shape[-1] != self.context_embed_dim:
                raise ValueError("context_embedding dimension mismatch")
        elif context_items is not None:
            if self.context_item_encoder is None:
                raise ValueError("context items supplied to a proxy without an item encoder")
            if context_items.shape[-1] != self.context_item_dim:
                raise ValueError(
                    f"context item dim must be {self.context_item_dim}; "
                    f"received {context_items.shape[-1]}"
                )
            item_embeddings = self.context_item_encoder(context_items)
            if context_mask is None:
                context_mask = torch.ones(
                    context_items.shape[:-1],
                    dtype=item_embeddings.dtype,
                    device=item_embeddings.device,
                )
            item_embeddings = item_embeddings * context_mask.unsqueeze(-1).to(
                item_embeddings.dtype
            )
            context_embedding = item_embeddings.sum(dim=-2)
        else:
            raw_set_summary = torch.cat(
                [z_core_excl_j, m_periph_excl_j], dim=-1
            )
            context_embedding = self.context_set_encoder(raw_set_summary)
        parts.append(context_embedding)

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
    Signed, propensity-augmented, multi-horizon ensemble proxy represented on GPU as
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

    "signed_policy_contrast"  (confirmatory H1)
        w = E_{a~pi}[f(s,a)] - E_{a~q}[f(s,a)], where q is uniform over
        candidate actions, pi is supplied via policy_probs_j, and the factual
        action is logged under behaviour b. Positive values preserve the
        operational meaning "the current neighbour policy is beneficial
        relative to random behaviour." This stochastic-policy contrast has a
        valid AIPW score; unlike a realised-action contrast, its target does
        not change with the randomly observed action.

    "range"  (Pieroth ICML 2024-style control baseline)
        w = max_{a valid} f(s,a) - min_{a valid} f(s,a), always >= 0

    "mean_abs"  (v1 form retained for before/after ablation)
        w = mean_{a != a_obs} |f(s,a) - f(s,a_obs)|

    ---------------------------------------------------------------------
    AUGMENTED INVERSE-PROPENSITY CONTRAST
    ---------------------------------------------------------------------
        tau_pi,q(s) = sum_a [pi(a|s)-q(a|s)] m(s,a)

        phi_DR = sum_a [pi(a|s)-q(a|s)] f_hat(s,a)
                 + [pi(A|s)-q(A|s)] / b(A|s)
                   * [R - f_hat(s,A)].

    Here ``q`` is the fixed uniform candidate-action reference and ``b`` is
    the logged epsilon-forcing behaviour policy. This is a fixed stochastic-
    policy value contrast, not the realised-action quantity
    ``psi(A)-E_pi[psi]``. The latter has a different residual coefficient and
    is intentionally not used by confirmatory H1. A row-level orthogonal
    score still requires a second-stage regression or aggregation for a
    conditional-effect claim. Inverse propensities are logged and clipping is
    measured explicitly to expose any variance/bias trade-off.
    """

    # Four valid modes.
    MODES = (
        "signed_aristocrat",
        "signed_oracle_matched",
        "signed_policy_contrast",
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
        discount: float = 0.97,
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
        context_item_dim: int = 0,
        debug_verbose: bool = False,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.core_dim = int(core_dim)
        self.periph_dim = int(periph_dim)
        self.belief_dim = int(belief_dim)
        # [FIX-X1] x_ij from Eq. 8; zero disables it for legacy behaviour.
        self.pair_feat_dim = int(pair_feat_dim)
        self.context_item_dim = int(context_item_dim)
        self.debug_verbose = bool(debug_verbose)
        self.n_ensemble = int(n_ensemble)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.buffer_size = int(buffer_size)
        self.device = device
        self.grad_clip = float(grad_clip)
        self.eps = float(eps)

        self.n_horizons = int(n_horizons)
        self.discount = float(discount)

        effect_mode = self.MODE_ALIASES.get(str(effect_mode), str(effect_mode))
        if effect_mode not in self.MODES:
            raise ValueError(
                f"effect_mode must be one of {self.MODES}; got {effect_mode!r}"
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
                    context_item_dim=self.context_item_dim,
                    hidden=self.hidden,
                    n_horizons=self.n_horizons,
                    discount=self.discount,
                    use_belief_input=self.use_belief_input,
                    dropout=float(ensemble_dropout),
                ).to(self.device)
            )

        self.use_vmap_ensemble = bool(use_vmap_ensemble) and _HAS_TORCH_FUNC
        self._compile_ensemble_flag = bool(compile_ensemble)

        # Retained for the grouped ContextBlock training path.  The vmap
        # optimizer remains authoritative normally; these optimizers are used
        # only after explicit stack/model synchronization.
        self.optims = [
            torch.optim.Adam(model.parameters(), lr=self.lr) for model in self.models
        ]

        if self.use_vmap_ensemble:
            self._setup_vmap_ensemble()

        # Independent RNG per member for independent weighted oversampling.
        self._member_rngs = [
            random.Random(int(seed) * 7919 + k) for k in range(self.n_ensemble)
        ]

        # ---------------------------------------------------------------
        # Fixed bootstrap membership is attached to immutable sample IDs.
        # redraw them on each call.
        #
        # If every train_step draws a new member pool, as the previous version
        # did to avoid Python scans, all members eventually see nearly the full
        # buffer and differ only in individual minibatches. The systematic
        # difference disappears, functions converge, and Eq. 10 sigma
        # collapses to zero as in v1.
        #
        # A deque rank is not sample identity: survivors move when old entries
        # are evicted.  Membership is therefore generated from (sample_id,
        # member seed) at insertion and stored with the sample.
        # ---------------------------------------------------------------
        self._bootstrap_seed = int(seed) * 104729 + 8191
        self._next_sample_id = 0

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
        # A DR-labelled experiment must prove that the correction path was
        # actually exercised.  These counters make a missing outcome or
        # propensity input visible instead of silently producing plug-in
        # scores under a DR configuration.
        self.latest_dr_applied = False
        self.latest_dr_applied_rows = 0
        self.latest_dr_clipped_rows = 0
        self.latest_dr_raw_inverse_max = 0.0
        self.total_dr_applied_calls = 0
        self.total_dr_applied_rows = 0
        self.total_dr_clipped_rows = 0

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

        def _fmodel(
            params, buffers, obs_i, a_i_oh, z, m, belief, pair_feat,
            context_items, context_mask,
        ):
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
                    context_items=context_items,
                    context_mask=context_mask,
                ),
            )

        # randomness="different" gives each member an independent dropout
        # mask if ensemble_dropout is enabled. vmap otherwise rejects random
        # operations without an explicit policy. [FIX-X1] also adds pair_feat
        # as the eighth _fmodel input dimension.
        self._vmap_forward_shared = vmap(
            _fmodel,
            in_dims=(0, 0, None, None, None, None, None, None, None, None),
            randomness="different",
        )
        context_in_dim = 0 if self.context_item_dim > 0 else None
        self._vmap_forward_per_member = vmap(
            _fmodel,
            in_dims=(0, 0, 0, 0, 0, 0, 0, 0, context_in_dim, context_in_dim),
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
        target_lag_rewards=None,
        target_lag_valid_mask=None,
        horizon_complete=None,
        behaviour_prob_j=None,
        was_forced=False,
        state_key=None,
        policy_probs_j=None,
        valid_action_mask=None,
        episode_id=None,
        timestep=None,
        context_items=None,
        context_mask=None,
        context_block=None,
        context_target_id=None,
    ):
        """
        Add one supervised sample.

        Args:
            target_lag_rewards:
                Direct rewards ``[r_t, ..., r_(t+H-1)]``. New collection code
                must provide this vector. ``target_returns_multi`` remains a
                compatibility alias for older callers and is interpreted as
                the already-migrated per-lag vector, not cumulative returns.
            behaviour_prob_j:
                b_j(a_j_obs | s) at collection time, required for DR. None
                disables DR for this sample and falls back to plug-in.
            was_forced:
                True when j's action was epsilon-forced, representing a true,
                higher-value intervention that is oversampled in training.
            state_key:
                Context identifier such as zone ID or coarse position hash.
        """
        if target_lag_rewards is not None and target_returns_multi is not None:
            raise ValueError(
                "Provide target_lag_rewards or target_returns_multi, not both"
            )
        lag_values = (
            target_lag_rewards
            if target_lag_rewards is not None
            else target_returns_multi
        )
        if lag_values is None:
            multi = np.full(
                (self.n_horizons,), float(target_return_h), dtype=np.float32
            )
        else:
            multi = np.asarray(lag_values, dtype=np.float32).reshape(-1)

            if multi.shape[0] != self.n_horizons:
                raise ValueError(
                    f"target_lag_rewards must have length {self.n_horizons}, "
                    f"got {multi.shape[0]}"
                )

        if target_lag_valid_mask is None:
            lag_valid = np.ones((self.n_horizons,), dtype=np.float32)
        else:
            lag_valid = np.asarray(target_lag_valid_mask, dtype=np.float32).reshape(-1)
            if lag_valid.shape[0] != self.n_horizons:
                raise ValueError("target_lag_valid_mask horizon mismatch")
            lag_valid = (lag_valid > 0.5).astype(np.float32)
        if horizon_complete is None:
            horizon_complete = bool(np.all(lag_valid > 0.5))

        sample_id = int(self._next_sample_id)
        self._next_sample_id += 1
        # Each member receives an independent SplitMix64 stream keyed by its
        # immutable sample ID. Linear congruential offsets are correlated and
        # can make all members select the same replay population.
        membership = np.asarray([
            (_splitmix64(
                sample_id ^ (self._bootstrap_seed + (member + 1) * 0x9E3779B97F4A7C15)
            ) / float(1 << 64)) < self.bootstrap_ratio
            for member in range(self.n_ensemble)
        ], dtype=bool)
        if not np.any(membership):
            membership[sample_id % self.n_ensemble] = True

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
            "target_lag_rewards": multi,                        # [n_horizons]
            "target_lag_valid_mask": lag_valid,
            "horizon_complete": bool(horizon_complete),
            "sample_id": sample_id,
            "bootstrap_membership": membership,
            "behaviour_prob_j": (
                None if behaviour_prob_j is None else float(behaviour_prob_j)
            ),
            "was_forced": bool(was_forced),
            "state_key": state_key,
            "policy_probs_j": (
                None
                if policy_probs_j is None
                else self._normalise_vector(policy_probs_j, self.action_dim)
            ),
            "valid_action_mask": (
                np.ones((self.action_dim,), dtype=bool)
                if valid_action_mask is None
                else np.asarray(valid_action_mask, dtype=bool).reshape(self.action_dim)
            ),
            "episode_id": episode_id,
            "timestep": timestep,
        }
        if self.context_item_dim > 0:
            if context_block is not None:
                sample["context_block"] = context_block
                sample["context_target_id"] = int(context_target_id)
            else:
                if context_items is None:
                    item_array = np.zeros(
                        (1, self.context_item_dim), dtype=np.float32
                    )
                    mask_array = np.zeros((1,), dtype=np.float32)
                else:
                    item_array = np.asarray(context_items, dtype=np.float32)
                    if item_array.ndim != 2 or item_array.shape[1] != self.context_item_dim:
                        raise ValueError(
                            "context_items must have shape [K, "
                            f"{self.context_item_dim}], received {item_array.shape}"
                        )
                    mask_array = (
                        np.ones((item_array.shape[0],), dtype=np.float32)
                        if context_mask is None
                        else np.asarray(context_mask, dtype=np.float32).reshape(-1)
                    )
                    if mask_array.shape != (item_array.shape[0],):
                        raise ValueError("context_mask must have one entry per context item")
                sample["context_items"] = item_array.copy()
                sample["context_mask"] = mask_array.copy()

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

    def get_action_coverage_diagnostics(self) -> Dict[str, object]:
        """Return action-head support measured at the replay boundary.

        Batch-level minimum frequencies are too noisy to validate overlap and
        can look healthy after resampling even when the underlying replay has
        an unsupported action.  This diagnostic operates on the complete
        retained buffer and reports factual and forced-only counts separately.
        """
        counts = np.zeros(self.action_dim, dtype=np.int64)
        forced_counts = np.zeros(self.action_dim, dtype=np.int64)

        for sample in self.buffer:
            action = int(sample["observed_action_j"])
            if 0 <= action < self.action_dim:
                counts[action] += 1
                if bool(sample.get("was_forced", False)):
                    forced_counts[action] += 1

        total = int(counts.sum())
        forced_total = int(forced_counts.sum())
        fractions = counts.astype(np.float64) / max(1, total)
        forced_fractions = forced_counts.astype(np.float64) / max(1, forced_total)

        return {
            "action_counts": counts.tolist(),
            "action_fractions": fractions.tolist(),
            "actions_seen": int(np.count_nonzero(counts)),
            "min_action_fraction": float(fractions.min()) if total else 0.0,
            "forced_action_counts": forced_counts.tolist(),
            "forced_action_fractions": forced_fractions.tolist(),
            "forced_actions_seen": int(np.count_nonzero(forced_counts)),
            "min_forced_action_fraction": (
                float(forced_fractions.min()) if forced_total else 0.0
            ),
            "n_samples": total,
            "n_forced_samples": forced_total,
        }

    # =====================================================================
    # Train
    # =====================================================================

    def _batch_to_tensors(self, batch, include_context=True):
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
            [b.get("target_lag_rewards", b.get("target_returns_multi")) for b in batch],
            axis=0,
        )                                                            # [B, n_horizons]
        target_valid = np.stack([
            b.get("target_lag_valid_mask", np.ones(self.n_horizons, dtype=np.float32))
            for b in batch
        ], axis=0)

        context_items_t = None
        context_mask_t = None
        if self.context_item_dim > 0 and include_context:
            item_rows = []
            item_masks = []
            for b in batch:
                if "context_block" in b:
                    block = b["context_block"]
                    raw = np.asarray(block["items"], dtype=np.float32)
                    ids = np.asarray(block["neighbor_ids"], dtype=np.int64)
                    keep = ids != int(b["context_target_id"])
                    item_rows.append(raw[keep])
                    item_masks.append(np.ones(int(np.count_nonzero(keep)), dtype=np.float32))
                else:
                    item_rows.append(np.asarray(b["context_items"], dtype=np.float32))
                    item_masks.append(np.asarray(b["context_mask"], dtype=np.float32))
            max_items = max(row.shape[0] for row in item_rows)
            padded_items = np.zeros(
                (len(batch), max_items, self.context_item_dim), dtype=np.float32
            )
            padded_mask = np.zeros((len(batch), max_items), dtype=np.float32)
            for index, (row, sample) in enumerate(zip(item_rows, batch)):
                padded_items[index, :row.shape[0]] = row
                mask = item_masks[index]
                padded_mask[index, :row.shape[0]] = mask
            context_items_t = torch.tensor(
                padded_items, dtype=torch.float32, device=self.device
            )
            context_mask_t = torch.tensor(
                padded_mask, dtype=torch.float32, device=self.device
            )

        return (
            torch.tensor(obs, dtype=torch.float32, device=self.device),
            self._one_hot(action_i),
            # a_j is now a raw int64 [B] index used for gather, not forward input.
            torch.tensor(action_j, dtype=torch.int64, device=self.device),
            torch.tensor(z, dtype=torch.float32, device=self.device),
            torch.tensor(m, dtype=torch.float32, device=self.device),
            torch.tensor(belief, dtype=torch.float32, device=self.device),
            torch.tensor(target_multi, dtype=torch.float32, device=self.device),
            torch.tensor(target_valid, dtype=torch.float32, device=self.device),
            torch.tensor(pair_feat, dtype=torch.float32, device=self.device),
            torch.tensor(b_obs, dtype=torch.float32, device=self.device),
            context_items_t,
            context_mask_t,
        )

    def _grouped_context_embeddings(self, model, batch):
        """Encode every replay ContextBlock once, then subtract each target.

        Pair replay records refer to a shared per-ego block.  Materialising
        and encoding an exclusion set for every target reintroduced O(N^3)
        work during training.  This routine keeps the literal DeepSets
        ``phi(X_i)`` / ``sum - phi(x_ij)`` construction for each model.
        """
        if self.context_item_dim <= 0:
            return None
        output = [None] * len(batch)
        grouped = {}
        for index, sample in enumerate(batch):
            block = sample.get("context_block")
            if block is None:
                grouped[("legacy", index)] = [index]
            else:
                grouped.setdefault(("block", id(block)), []).append(index)
        for key, indices in grouped.items():
            sample = batch[indices[0]]
            block = sample.get("context_block")
            if block is None:
                items = np.asarray(sample["context_items"], dtype=np.float32)
                encoded = model.context_item_encoder(torch.as_tensor(
                    items, dtype=torch.float32, device=self.device
                )).sum(dim=0)
                output[indices[0]] = encoded
                continue
            raw = np.asarray(block["items"], dtype=np.float32)
            ids = np.asarray(block["neighbor_ids"], dtype=np.int64)
            encoded = model.context_item_encoder(torch.as_tensor(
                raw, dtype=torch.float32, device=self.device
            ))
            total = encoded.sum(dim=0)
            positions = {int(agent): pos for pos, agent in enumerate(ids.tolist())}
            for index in indices:
                target = int(batch[index]["context_target_id"])
                if target not in positions:
                    raise ValueError("ContextBlock does not contain its target neighbor")
                output[index] = total - encoded[positions[target]]
        return torch.stack(output, dim=0)

    def _discounted_return_residual(self, prediction, target, valid_mask=None):
        """Absolute error of the discounted H-step return for each row."""
        if prediction.shape != target.shape or prediction.shape[-1] != self.n_horizons:
            raise ValueError(
                "proxy residual horizon mismatch: "
                f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}, "
                f"expected H={self.n_horizons}"
            )
        weights = torch.pow(
            torch.as_tensor(
                self.discount, dtype=prediction.dtype, device=prediction.device
            ),
            torch.arange(
                self.n_horizons,
                dtype=prediction.dtype,
                device=prediction.device,
            ),
        )
        if valid_mask is None:
            valid_mask = torch.ones_like(target)
        valid_mask = valid_mask.to(dtype=prediction.dtype, device=prediction.device)
        return torch.abs(
            torch.sum(prediction * weights * valid_mask, dim=-1)
            - torch.sum(target * weights * valid_mask, dim=-1)
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

        pool_positions = np.asarray([
            index for index, sample in enumerate(buf_list)
            if bool(sample.get("bootstrap_membership", [True] * self.n_ensemble)[member_idx])
        ], dtype=np.int64)

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

        # ContextBlock replay is grouped by block identity so each member
        # encodes phi(X_i) once and derives all target exclusions by
        # subtraction. Synchronize explicitly around the compatibility update
        # rather than letting the vmap path repeat nonlinear set encodes.
        if self.use_vmap_ensemble and any(
            "context_block" in sample for sample in self.buffer
        ):
            self._sync_stacked_to_models()
            value = self._train_step_fallback(n_steps, batch_size, holdout_size)
            self._sync_models_to_stacked()
            return value
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

            obs_l, ai_l, aj_l, z_l, m_l, bl_l, tgt_l, valid_l = [], [], [], [], [], [], [], []
            pf_l = []   # [FIX-X1] x_ij per member
            bobs_l = []  # [FIX-HC1] b_j(a_j|s) per member
            ctx_l, ctx_mask_l = [], []

            for b in member_batches:
                (obs_t, a_i_oh, a_j_idx, z_t, m_t, belief_t, target_multi_t,
                 target_valid_t, pf_t, bobs_t, context_items_t, context_mask_t) = (
                    self._batch_to_tensors(b)
                )
                pf_l.append(pf_t); bobs_l.append(bobs_t)
                ctx_l.append(context_items_t); ctx_mask_l.append(context_mask_t)
                obs_l.append(obs_t)
                ai_l.append(a_i_oh)
                aj_l.append(a_j_idx)
                z_l.append(z_t)
                m_l.append(m_t)
                bl_l.append(belief_t)
                tgt_l.append(target_multi_t)
                valid_l.append(target_valid_t)

            obs_e = torch.stack(obs_l, dim=0)    # [E, B, obs_dim]
            ai_e = torch.stack(ai_l, dim=0)      # [E, B, A]
            aj_e = torch.stack(aj_l, dim=0)      # [E, B]
            z_e = torch.stack(z_l, dim=0)        # [E, B, core_dim]
            m_e = torch.stack(m_l, dim=0)        # [E, B, periph_dim]
            bel_e = torch.stack(bl_l, dim=0)     # [E, B, belief_dim]
            tgt_e = torch.stack(tgt_l, dim=0)    # [E, B, H]
            target_valid_e = torch.stack(valid_l, dim=0)  # [E, B, H]
            pf_e = torch.stack(pf_l, dim=0)      # [E, B, pair_feat_dim]
            bobs_e = torch.stack(bobs_l, dim=0)  # [E, B]
            ctx_e = (
                torch.stack(ctx_l, dim=0) if self.context_item_dim > 0 else None
            )
            ctx_mask_e = (
                torch.stack(ctx_mask_l, dim=0)
                if self.context_item_dim > 0 else None
            )

            # One vmap operation runs n_ensemble forwards in parallel on GPU.
            preds_all = self._vmap_forward_per_member(
                self._stacked_params, self._stacked_buffers,
                obs_e, ai_e, z_e, m_e, bel_e, pf_e, ctx_e, ctx_mask_e,
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

            sq_raw = F.mse_loss(preds, tgt_e, reduction="none")
            sq = (sq_raw * target_valid_e).sum(dim=2) / torch.clamp(
                target_valid_e.sum(dim=2), min=1.0
            )
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
                    self._discounted_return_residual(preds, tgt_e, target_valid_e), dim=1
                )  # [E] discounted H-return residual.
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
                if self.debug_verbose:
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

            (ho_obs, ho_ai, ho_aj, ho_z, ho_m, ho_b, ho_target, ho_valid, ho_pf,
             _ho_bobs, ho_ctx, ho_ctx_mask) = (
                self._batch_to_tensors(ho_batch)
            )

            self._ensemble_train_mode(False)

            with torch.no_grad():
                stacked_all = self._vmap_forward_shared(
                    self._stacked_params, self._stacked_buffers,
                    ho_obs, ho_ai, ho_z, ho_m, ho_b, ho_pf,
                    ho_ctx, ho_ctx_mask,
                )  # [E,B,A,H] from one vmap over data shared by all members.

                E_, B_, A_, H_ = stacked_all.shape
                ho_idx = ho_aj.view(1, B_, 1, 1).expand(E_, B_, 1, H_)
                stacked = torch.gather(stacked_all, dim=2, index=ho_idx).squeeze(2)  # [E, B, H]
                pred_mean = stacked.mean(dim=0)  # [B, H]

                holdout_residual_t = torch.mean(
                    self._discounted_return_residual(pred_mean, ho_target, ho_valid)
                )

                # [L3] Diagnose genuine ensemble disagreement. A value near
                # zero indicates a collapsed ensemble and meaningless sigma.
                if stacked.shape[0] > 1:
                    self.latest_ensemble_disagreement = float(
                        torch.mean(torch.std(stacked, dim=0)).item()
                    )  # One synchronization for the complete holdout evaluation.

        if len(per_step_losses) == 0:
            if self.debug_verbose:
                print("[TRAIN-DEBUG] all training steps were skipped")
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
                 target_valid_t, pf_t, _bobs_t, context_items_t, context_mask_t) = (
                    self._batch_to_tensors(batch, include_context=False)
                )
                context_embedding_t = self._grouped_context_embeddings(model, batch)

                model.train()

                pred_all = model(
                    obs_i=obs_t,
                    action_i_onehot=a_i_oh,
                    z_core_excl_j=z_t,
                    m_periph_excl_j=m_t,
                    belief_summary=belief_t,
                    pair_feat=pf_t,
                    context_items=context_items_t,
                    context_mask=context_mask_t,
                    context_embedding=context_embedding_t,
                )

                B_, A_, H_ = pred_all.shape
                idx = a_j_idx.view(B_, 1, 1).expand(B_, 1, H_)
                pred = torch.gather(pred_all, dim=1, index=idx).squeeze(1)  # [B, H]

                loss = ((pred - target_multi_t).pow(2) * target_valid_t).sum() / torch.clamp(
                    target_valid_t.sum(), min=1.0
                )

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optim.step()

                all_losses.append(loss.detach())

                with torch.no_grad():
                    res = torch.mean(
                        self._discounted_return_residual(
                            pred, target_multi_t, target_valid_t
                        )
                    )
                train_residuals.append(res)

            self.last_train_batch_count += 1

        holdout_residual = None

        if holdout_size > 0 and len(self.buffer) > holdout_size:
            ho_batch = random.sample(list(self.buffer), int(holdout_size))
            (ho_obs, ho_ai, ho_aj, ho_z, ho_m, ho_b, ho_target, ho_valid, ho_pf,
             _ho_bobs2, ho_ctx, ho_ctx_mask) = (
                self._batch_to_tensors(ho_batch, include_context=False)
            )

            with torch.no_grad():
                preds = []
                for model in self.models:
                    model.eval()
                    ho_context_embedding = self._grouped_context_embeddings(
                        model, ho_batch
                    )
                    pred_all = model(
                        obs_i=ho_obs, action_i_onehot=ho_ai,
                        z_core_excl_j=ho_z, m_periph_excl_j=ho_m,
                        belief_summary=ho_b, pair_feat=ho_pf,
                        context_items=ho_ctx, context_mask=ho_ctx_mask,
                        context_embedding=ho_context_embedding,
                    )
                    B_, A_, H_ = pred_all.shape
                    idx = ho_aj.view(B_, 1, 1).expand(B_, 1, H_)
                    preds.append(torch.gather(pred_all, dim=1, index=idx).squeeze(1))

                stacked = torch.stack(preds, dim=0)
                pred_mean = stacked.mean(dim=0)
                holdout_residual = float(torch.mean(
                    self._discounted_return_residual(pred_mean, ho_target, ho_valid)
                ))


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
        context_items=None,
        context_mask=None,
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
        ctx_t, ctx_mask_t = self._context_item_tensors(
            context_items, context_mask, B
        )

        self._ensemble_train_mode(False)

        with torch.no_grad():
            if self.use_vmap_ensemble:
                out = self._vmap_forward_shared(
                    self._stacked_params, self._stacked_buffers,
                    obs_t, a_i_oh, z_t, m_t, belief_t, pf_t,
                    ctx_t, ctx_mask_t,
                )  # [E, B, A, n_horizons]
            else:
                out = self._predict_all_actions_loop(
                    obs_t, a_i_oh, z_t, m_t, belief_t, pf_t,
                    ctx_t, ctx_mask_t)

        return out  # [E, B, A, n_horizons]

    def _predict_all_actions_loop(
        self, obs, a_i_oh, z, m, belief, pair_feat=None,
        context_items=None, context_mask=None,
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
                context_items=context_items, context_mask=context_mask,
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

    def _sync_models_to_stacked(self):
        """Publish grouped per-model updates back to the vmap parameter stack."""
        if not self.use_vmap_ensemble:
            return
        params, buffers = stack_module_state(self.models)
        with torch.no_grad():
            for name, value in params.items():
                self._stacked_params[name].copy_(value)
            for name, value in buffers.items():
                self._stacked_buffers[name].copy_(value)
        # Moment estimates belong to the prior stacked trajectory and cannot
        # be safely reused after a separately optimized grouped update.
        self.optim.state.clear()

    def _predict_all_actions_reference(
        self, obs, action_i, z, m, belief, pair_feat=None,
        context_items=None, context_mask=None,
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
        ctx_t, ctx_mask_t = self._context_item_tensors(
            context_items, context_mask, int(obs_t.shape[0])
        )

        with torch.no_grad():
            out = self._predict_all_actions_loop(
                obs_t, a_i_oh, z_t, m_t, belief_t, pf_t,
                ctx_t, ctx_mask_t)

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
                f"pair_feat_dim={self.pair_feat_dim}, but the call site did "
                "not provide x_ij"
            )
        arr = np.asarray(pair_feat, dtype=np.float32).reshape(B, -1)
        if arr.shape[1] != self.pair_feat_dim:
            raise ValueError(
                f"x_ij must have {self.pair_feat_dim} dimensions; got {arr.shape[1]}"
            )
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _context_item_tensors(self, context_items, context_mask, batch_size):
        if self.context_item_dim <= 0 or context_items is None:
            return None, None
        items = np.asarray(context_items, dtype=np.float32)
        if (
            items.ndim != 3
            or items.shape[0] != int(batch_size)
            or items.shape[2] != self.context_item_dim
        ):
            raise ValueError(
                "context_items must have shape "
                f"[{batch_size}, K, {self.context_item_dim}], got {items.shape}"
            )
        if context_mask is None:
            mask = np.ones(items.shape[:2], dtype=np.float32)
        else:
            mask = np.asarray(context_mask, dtype=np.float32)
            if mask.shape != items.shape[:2]:
                raise ValueError(
                    f"context_mask must have shape {items.shape[:2]}, got {mask.shape}"
                )
        return (
            torch.tensor(items, dtype=torch.float32, device=self.device),
            torch.tensor(mask, dtype=torch.float32, device=self.device),
        )

    # =====================================================================
    # Effect computation.
    # =====================================================================

    def _compute_effects(
        self,
        preds_all: torch.Tensor,        # [E, B, A, n_horizons]
        action_j_obs: np.ndarray,       # [B]
        policy_probs_j: Optional[np.ndarray] = None,   # [B, A]
        valid_action_mask: Optional[np.ndarray] = None,  # [B, A]
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
        target_prob_obs = None

        if valid_action_mask is None:
            valid = torch.ones(B, A, dtype=torch.bool, device=self.device)
        else:
            valid_arr = np.asarray(valid_action_mask, dtype=bool)
            if valid_arr.shape != (B, A) or np.any(valid_arr.sum(axis=1) == 0):
                raise ValueError(
                    "valid_action_mask must have shape "
                    f"{(B, A)} and retain at least one action per row"
                )
            valid = torch.tensor(valid_arr, dtype=torch.bool, device=self.device)

        def _normalize_over_valid(weights):
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            row_sum = weights.sum(dim=1, keepdim=True)
            fallback = valid.to(weights.dtype)
            fallback = fallback / fallback.sum(dim=1, keepdim=True)
            return torch.where(
                row_sum > self.eps,
                weights / torch.clamp(row_sum, min=self.eps),
                fallback,
            )

        idx_obs = torch.tensor(
            np.asarray(action_j_obs, dtype=np.int64).reshape(-1),
            dtype=torch.long,
            device=self.device,
        ).clamp(0, A - 1)  # [B]

        idx_exp = (
            idx_obs.view(1, B, 1, 1).expand(E, B, 1, H)
        )  # [E, B, 1, H]
        if not bool(torch.all(torch.gather(valid, 1, idx_obs.view(B, 1)))):
            raise ValueError("observed_action_j contains an invalid action")

        f_obs = torch.gather(preds_all, dim=2, index=idx_exp).squeeze(2)
        # f_obs: [E, B, H]

        dr_correction = torch.zeros(B, dtype=torch.float32, device=self.device)

        # ---------------------------------------------------------------
        if mode == "signed_aristocrat":
            if policy_probs_j is None:
                w = valid.to(torch.float32)
            else:
                w_arr = np.asarray(policy_probs_j, dtype=np.float32)
                if w_arr.shape != (B, A):
                    raise ValueError(
                        f"policy_probs_j must have shape {(B, A)}, received "
                        f"{w_arr.shape}"
                    )
                if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0):
                    raise ValueError("policy_probs_j contains invalid probabilities")
                w = torch.tensor(
                    w_arr,
                    dtype=torch.float32,
                    device=self.device,
                )  # [B, A]
            w = _normalize_over_valid(w)

            baseline = torch.einsum("ebah,ba->ebh", preds_all, w)
            effect_per_h = f_obs - baseline  # [E, B, H]
            target_prob_obs = torch.gather(w, 1, idx_obs.view(B, 1)).squeeze(1)

        elif mode == "signed_oracle_matched":
            cand = torch.tensor(
                [a for a in self.candidate_actions if 0 <= a < A],
                dtype=torch.long,
                device=self.device,
            )  # [n_cand]

            if cand.numel() == 0:
                cand = torch.arange(A, dtype=torch.long, device=self.device)

            candidate_mask = torch.zeros(A, dtype=torch.bool, device=self.device)
            candidate_mask[cand] = True
            row_candidates = valid & candidate_mask.view(1, A)
            empty = row_candidates.sum(dim=1) == 0
            row_candidates[empty] = valid[empty]
            q = row_candidates.to(torch.float32)
            q = q / q.sum(dim=1, keepdim=True)
            cand_mean = torch.einsum("ebah,ba->ebh", preds_all, q)

            effect_per_h = cand_mean - f_obs           # [E, B, H]
            # q(a_obs|s) for the uniform candidate-action intervention.
            # It is zero when the factual action is outside the candidate set.
            target_prob_obs = torch.gather(q, 1, idx_obs.view(B, 1)).squeeze(1)

        elif mode == "signed_policy_contrast":
            if policy_probs_j is None:
                raise ValueError(
                    "signed_policy_contrast requires the complete target "
                    "policy distribution in policy_probs_j"
                )
            pi_full_arr = np.asarray(policy_probs_j, dtype=np.float32)
            if pi_full_arr.shape != (B, A):
                raise ValueError(
                    f"policy_probs_j must have shape {(B, A)}, received "
                    f"{pi_full_arr.shape}"
                )
            if (
                not np.all(np.isfinite(pi_full_arr))
                or np.any(pi_full_arr < 0.0)
            ):
                raise ValueError("target policy distribution is invalid")
            pi_full = torch.tensor(
                pi_full_arr, dtype=torch.float32, device=self.device,
            )
            pi_full = _normalize_over_valid(pi_full)
            cand = torch.tensor(
                [a for a in self.candidate_actions if 0 <= a < A],
                dtype=torch.long,
                device=self.device,
            )
            if cand.numel() == 0:
                cand = torch.arange(A, dtype=torch.long, device=self.device)
            candidate_mask = torch.zeros(A, dtype=torch.bool, device=self.device)
            candidate_mask[cand] = True
            row_candidates = valid & candidate_mask.view(1, A)
            empty = row_candidates.sum(dim=1) == 0
            row_candidates[empty] = valid[empty]
            q = row_candidates.to(torch.float32)
            q = q / q.sum(dim=1, keepdim=True)
            q_mean = torch.einsum("ebah,ba->ebh", preds_all, q)
            pi_mean = torch.einsum("ebah,ba->ebh", preds_all, pi_full)
            effect_per_h = pi_mean - q_mean
            target_prob_obs = torch.gather(q, 1, idx_obs.view(B, 1)).squeeze(1)
            policy_prob_obs = torch.gather(
                pi_full, 1, idx_obs.view(B, 1)
            ).squeeze(1)

        elif mode == "range":
            expanded_valid = valid.view(1, B, A, 1)
            max_f = preds_all.masked_fill(~expanded_valid, -torch.inf).max(dim=2).values
            min_f = preds_all.masked_fill(~expanded_valid, torch.inf).min(dim=2).values
            effect_per_h = max_f - min_f               # [E, B, H]

        elif mode == "mean_abs":
            diff = torch.abs(preds_all - f_obs.unsqueeze(2))  # [E, B, A, H]

            mask = valid.to(torch.float32)
            mask.scatter_(1, idx_obs.view(B, 1), 0.0)          # [B, A]

            denom = torch.clamp(mask.sum(dim=1), min=1.0)      # [B]

            effect_per_h = (
                torch.einsum("ebah,ba->ebh", diff, mask) / denom.view(1, B, 1)
            )  # [E, B, H]

        else:
            raise ValueError(f"invalid effect mode: {mode}")

        # ---------------------------------------------------------------
        # AIPW applies only to the fixed stochastic-policy contrast. The two
        # realised-action modes do not have a valid row-level DR correction.
        # ---------------------------------------------------------------
        apply_dr = (
            self.use_doubly_robust
            and mode == "signed_policy_contrast"
            and observed_returns is not None
            and behaviour_probs_obs is not None
        )

        dr_applied = False
        dr_applied_rows = 0
        dr_clipped_rows = 0
        dr_raw_inverse_max = 0.0
        dr_weight = torch.zeros(B, dtype=torch.float32, device=self.device)

        if apply_dr:
            returns_arr = np.asarray(observed_returns, dtype=np.float32).reshape(-1)
            propensity_arr = np.asarray(
                behaviour_probs_obs, dtype=np.float32
            ).reshape(-1)
            if returns_arr.shape != (B,) or propensity_arr.shape != (B,):
                raise ValueError(
                    "DR inputs must contain one return and one observed-action "
                    f"propensity per row; received {returns_arr.shape} and "
                    f"{propensity_arr.shape}, expected {(B,)}"
                )
            if not np.all(np.isfinite(returns_arr)):
                raise ValueError("observed_returns contains NaN or infinity")
            if (
                not np.all(np.isfinite(propensity_arr))
                or np.any(propensity_arr <= 0.0)
                or np.any(propensity_arr > 1.0 + 1e-6)
            ):
                raise ValueError(
                    "behaviour_probs_obs must be finite and in (0, 1]"
                )

            R_obs = torch.tensor(
                returns_arr,
                dtype=torch.float32,
                device=self.device,
            )  # [B]

            b_obs = torch.tensor(
                propensity_arr,
                dtype=torch.float32,
                device=self.device,
            ).clamp(min=self.eps, max=1.0)  # [B]

            residual = R_obs.view(1, B) - f_obs[:, :, -1]  # [E, B]

            # For V_pi(s)-V_q(s), sampled under logging policy b, the AIPW
            # score is
            #   sum_a (pi(a)-q(a)) f(a)
            #   + ((pi(A)-q(A))/b(A)) * (Y-f(A)).
            # This is an aggregate stochastic-policy contrast. A row-specific
            # target such as Q(A)-V_pi changes with random A and cannot acquire
            # conditional double robustness by attaching a residual to that
            # same row; those legacy modes therefore remain plug-in only.
            raw_inv_b = 1.0 / b_obs
            dr_clipped_rows = int(
                torch.sum(raw_inv_b > self.iw_clip).detach().cpu().item()
            )
            dr_raw_inverse_max = float(
                torch.max(raw_inv_b).detach().cpu().item()
            )
            inv_b = torch.clamp(raw_inv_b, max=self.iw_clip)
            contrast_weight = (
                policy_prob_obs - target_prob_obs
            ) * inv_b

            correction = residual * contrast_weight.view(1, B)  # [E, B]

            # The residual correction is defined for the terminal return, so
            # earlier horizons remain plug-in values. Keep that pure plug-in
            # vector for latency diagnostics rather than mixing estimands
            # across horizons.
            effect_per_h_plugin = effect_per_h
            effect_per_h = effect_per_h.clone()
            effect_per_h[:, :, -1] = effect_per_h[:, :, -1] + correction

            dr_correction = torch.mean(torch.abs(correction), dim=0)  # [B]
            dr_weight = contrast_weight
            dr_applied = True
            dr_applied_rows = B

        return {
            "effect": effect_per_h[:, :, -1],   # [E, B]
            "effect_per_h": effect_per_h,       # [E,B,H], with DR at h=H
            # [FIX-DR-H] Consistent estimator across horizons for latency diagnostics.
            "effect_per_h_plugin": (
                effect_per_h_plugin if apply_dr else effect_per_h
            ),
            "dr_correction": dr_correction,     # [B]
            "dr_weight": dr_weight,             # [B]
            "dr_applied": bool(dr_applied),
            "dr_applied_rows": int(dr_applied_rows),
            "dr_clipped_rows": int(dr_clipped_rows),
            "dr_raw_inverse_max": float(dr_raw_inverse_max),
        }

    def _cumulative_from_lag(self, lag_predictions: torch.Tensor) -> torch.Tensor:
        """Convert direct lag responses into discounted cumulative Q heads."""
        horizon = int(lag_predictions.shape[-1])
        discounts = torch.pow(
            torch.as_tensor(
                self.discount,
                dtype=lag_predictions.dtype,
                device=lag_predictions.device,
            ),
            torch.arange(
                horizon,
                dtype=lag_predictions.dtype,
                device=lag_predictions.device,
            ),
        )
        return torch.cumsum(lag_predictions * discounts, dim=-1)

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
        context_items_batch=None,
        context_mask_batch=None,
        valid_action_mask_batch=None,
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
            context_items_batch=context_items_batch,
            context_mask_batch=context_mask_batch,
            valid_action_mask_batch=valid_action_mask_batch,
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
        context_items_batch=None,
        context_mask_batch=None,
        valid_action_mask_batch=None,
        lag_preds_all_override=None,
    ) -> Dict[str, np.ndarray]:
        """
        Full version providing every field needed by influence_signature.py.

        Returns dict of np.ndarray:
            d_mu/d_sigma  [B] directional contrast and its uncertainty
            c_mu/c_sigma  [B] structural capacity and its uncertainty
            mu/sigma      legacy aliases for d_mu/d_sigma
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
                "d_mu": z,
                "d_sigma": z,
                "c_mu": z,
                "c_sigma": z,
                "mu_per_h": np.zeros((0, self.n_horizons), dtype=np.float32),
                "q_mu": np.zeros((0, self.action_dim), dtype=np.float32),
                "q_sigma": np.zeros((0, self.action_dim), dtype=np.float32),
                "g_lag_mu": np.zeros(
                    (0, self.action_dim, self.n_horizons), dtype=np.float32
                ),
                "g_lag_sigma": np.zeros(
                    (0, self.action_dim, self.n_horizons), dtype=np.float32
                ),
                "c_lag_mu": np.zeros((0, self.n_horizons), dtype=np.float32),
                "c_lag_sigma": np.zeros((0, self.n_horizons), dtype=np.float32),
                "latency_center": z,
                "latency_onset": np.zeros((0,), dtype=np.int64),
                "latency_peak": np.zeros((0,), dtype=np.int64),
                "latency_valid": z,
                "mu_range": z,
                "dr_correction": z,
                "dr_weight": z,
                "dr_applied": False,
                "dr_applied_rows": 0,
                "dr_clipped_rows": 0,
                "dr_raw_inverse_max": 0.0,
                "valid_action_mask": np.zeros(
                    (0, self.action_dim), dtype=np.float32
                ),
            }

        obs = np.asarray(obs_i_batch, dtype=np.float32)
        z_arr = np.asarray(z_core_excl_j_batch, dtype=np.float32)
        m_arr = np.asarray(m_periph_excl_j_batch, dtype=np.float32)
        belief = np.asarray(belief_summary_batch, dtype=np.float32)
        a_i = np.asarray(action_i_batch, dtype=np.int64).reshape(-1)
        a_j = np.asarray(observed_action_j_batch, dtype=np.int64).reshape(-1)
        if valid_action_mask_batch is None:
            valid_action_mask = np.ones(
                (B, self.action_dim), dtype=bool
            )
        else:
            valid_action_mask = np.asarray(
                valid_action_mask_batch, dtype=bool
            )
            if (
                valid_action_mask.shape != (B, self.action_dim)
                or np.any(valid_action_mask.sum(axis=1) == 0)
            ):
                raise ValueError(
                    "valid_action_mask_batch must have shape "
                    f"{(B, self.action_dim)} and retain one action per row"
                )

        # One forward pass over both alternative actions and ensemble members.
        lag_preds_all = lag_preds_all_override
        if lag_preds_all is None:
            lag_preds_all = self._predict_all_actions(
                obs=obs, action_i=a_i, z=z_arr, m=m_arr, belief=belief,
                pair_feat=pair_feat_batch,
                context_items=context_items_batch,
                context_mask=context_mask_batch,
            )
        preds_all = self._cumulative_from_lag(lag_preds_all)

        diagnostic_res = self._compute_effects(
            preds_all=preds_all,
            action_j_obs=a_j,
            policy_probs_j=policy_probs_j_batch,
            valid_action_mask=valid_action_mask,
            observed_returns=observed_returns_batch,
            behaviour_probs_obs=behaviour_probs_obs_batch,
            mode=self.effect_mode,
        )
        # The online directional signal is deliberately the lower-variance
        # plug-in estimate. Row-level AIPW is exported as a diagnostic and is
        # never silently routed into signatures, memory, or pair supervision.
        plugin_res = self._compute_effects(
            preds_all=preds_all,
            action_j_obs=a_j,
            policy_probs_j=policy_probs_j_batch,
            valid_action_mask=valid_action_mask,
            mode=self.effect_mode,
        )

        effect = plugin_res["effect"]              # [E, B]
        effect_per_h = plugin_res["effect_per_h"]  # [E, B, H]
        diagnostic_effect = diagnostic_res["effect"]

        mu = torch.mean(effect, dim=0)      # [B]

        if effect.shape[0] <= 1:
            sigma = torch.zeros_like(mu)
        else:
            sigma = torch.sqrt(
                torch.var(effect, dim=0, unbiased=True) + self.eps
            )  # [B]

        mu_per_h = torch.mean(effect_per_h, dim=0)  # [B, H]

        diagnostic_mu = torch.mean(diagnostic_effect, dim=0)
        diagnostic_sigma = (
            torch.zeros_like(diagnostic_mu)
            if diagnostic_effect.shape[0] <= 1
            else torch.sqrt(
                torch.var(diagnostic_effect, dim=0, unbiased=True) + self.eps
            )
        )

        # [SIG-5D] The latency component was removed, reducing the signature to
        # R^5; see influence_signature.py. mu_per_h remains only as a raw
        # horizon diagnostic and carries no latency-centroid claim.

        # Capacity C is always the per-action response range, independent of
        # the selected diagnostic effect mode.  It is the only quantity passed
        # to the structural belief/core selector.
        res_range = self._compute_effects(
            preds_all=preds_all,
            action_j_obs=a_j,
            valid_action_mask=valid_action_mask,
            mode="range",
        )
        c_effect = res_range["effect"]
        c_mu = torch.mean(c_effect, dim=0)                         # [B]
        c_sigma = (
            torch.zeros_like(c_mu)
            if c_effect.shape[0] <= 1
            else torch.sqrt(torch.var(c_effect, dim=0, unbiased=True) + self.eps)
        )

        # q(a)=E_member[Q(a)] is exported for H1a response-surface calibration
        # and oracle diagnostics.  The runner does not use it as a graph input.
        q_terminal = preds_all[:, :, :, -1]
        q_mu = torch.mean(q_terminal, dim=0)                       # [B,A]
        q_sigma = (
            torch.zeros_like(q_mu)
            if q_terminal.shape[0] <= 1
            else torch.sqrt(torch.var(q_terminal, dim=0, unbiased=True) + self.eps)
        )

        # Direct lag-response and capacity spectra support the gated latency
        # experiment without selecting one action for all lags.
        g_mu = torch.mean(lag_preds_all, dim=0)  # [B,A,H]
        g_sigma = (
            torch.zeros_like(g_mu)
            if lag_preds_all.shape[0] <= 1
            else torch.sqrt(
                torch.var(lag_preds_all, dim=0, unbiased=True) + self.eps
            )
        )
        valid_lag = torch.tensor(
            valid_action_mask, dtype=torch.bool, device=self.device
        ).view(1, B, self.action_dim, 1)
        c_lag_members = (
            lag_preds_all.masked_fill(~valid_lag, -torch.inf).max(dim=2).values
            - lag_preds_all.masked_fill(~valid_lag, torch.inf).min(dim=2).values
        )  # [E,B,H]
        c_lag_mu = torch.mean(c_lag_members, dim=0)
        c_lag_sigma = (
            torch.zeros_like(c_lag_mu)
            if c_lag_members.shape[0] <= 1
            else torch.sqrt(
                torch.var(c_lag_members, dim=0, unbiased=True) + self.eps
            )
        )
        lag_mass = torch.clamp(c_lag_mu, min=0.0)
        lag_total = lag_mass.sum(dim=1)
        lag_axis = torch.arange(
            self.n_horizons, dtype=lag_mass.dtype, device=lag_mass.device
        ).view(1, -1)
        latency_center = (
            (lag_mass * lag_axis).sum(dim=1)
            / torch.clamp(lag_total, min=self.eps)
        )
        latency_valid = lag_total > self.eps
        latency_center = torch.where(
            latency_valid, latency_center, torch.full_like(latency_center, -1.0)
        )
        peak_value, latency_peak = lag_mass.max(dim=1)
        latency_peak = torch.where(
            latency_valid, latency_peak, torch.full_like(latency_peak, -1)
        )
        onset_mask = lag_mass >= (0.05 * peak_value).unsqueeze(1)
        latency_onset = torch.argmax(onset_mask.to(torch.int64), dim=1)
        latency_onset = torch.where(
            latency_valid, latency_onset, torch.full_like(latency_onset, -1)
        )

        self.latest_dr_correction_magnitude = float(
            torch.mean(diagnostic_res["dr_correction"]).item()
        )
        self.latest_dr_applied = bool(diagnostic_res["dr_applied"])
        self.latest_dr_applied_rows = int(diagnostic_res["dr_applied_rows"])
        self.latest_dr_clipped_rows = int(diagnostic_res["dr_clipped_rows"])
        self.latest_dr_raw_inverse_max = float(diagnostic_res["dr_raw_inverse_max"])
        if self.latest_dr_applied:
            self.total_dr_applied_calls += 1
            self.total_dr_applied_rows += self.latest_dr_applied_rows
            self.total_dr_clipped_rows += self.latest_dr_clipped_rows

        # One .cpu().numpy() per output at the API boundary. Leaving GPU is
        # required here because downstream influence_signature.py and
        # belief_layer.py still accept NumPy and were not vectorized in this
        # revision; see README_INTEGRATION.md.
        to_np = lambda t: t.detach().cpu().numpy().astype(np.float32)

        return {
            "mu": to_np(mu),
            "sigma": to_np(sigma),
            "d_mu": to_np(mu),
            "d_sigma": to_np(sigma),
            "d_plugin_mu": to_np(mu),
            "d_plugin_sigma": to_np(sigma),
            "d_row_aipw_mu": to_np(diagnostic_mu),
            "d_row_aipw_sigma": to_np(diagnostic_sigma),
            "c_mu": to_np(c_mu),
            "c_sigma": to_np(c_sigma),
            "mu_per_h": to_np(mu_per_h),
            "mu_range": to_np(c_mu),
            "q_mu": to_np(q_mu),
            "q_sigma": to_np(q_sigma),
            "g_lag_mu": to_np(g_mu),
            "g_lag_sigma": to_np(g_sigma),
            "c_lag_mu": to_np(c_lag_mu),
            "c_lag_sigma": to_np(c_lag_sigma),
            "latency_center": to_np(latency_center),
            "latency_onset": to_np(latency_onset),
            "latency_peak": to_np(latency_peak),
            "latency_valid": to_np(latency_valid.to(torch.float32)),
            "dr_correction": to_np(diagnostic_res["dr_correction"]),
            "dr_weight": to_np(diagnostic_res["dr_weight"]),
            "dr_applied": bool(diagnostic_res["dr_applied"]),
            "dr_applied_rows": int(diagnostic_res["dr_applied_rows"]),
            "dr_clipped_rows": int(diagnostic_res["dr_clipped_rows"]),
            "dr_raw_inverse_max": float(diagnostic_res["dr_raw_inverse_max"]),
            "valid_action_mask": valid_action_mask.astype(np.float32),
        }

    def score_batch_from_context_block(self, *, context_block, target_ids, **kwargs):
        """Encode one ego's context once per ensemble member, then subtract.

        This is the literal DeepSets ``S_i-e_ij`` path used by graph scoring.
        It removes repeated nonlinear context encoding across all targets.
        """
        if self.context_item_dim <= 0:
            return self.score_batch_full(**kwargs)
        ids = np.asarray(context_block["neighbor_ids"], dtype=np.int64)
        items = np.asarray(context_block["items"], dtype=np.float32)
        targets = np.asarray(target_ids, dtype=np.int64)
        positions = {int(agent): index for index, agent in enumerate(ids.tolist())}
        if (items.ndim != 2 or items.shape[1] != self.context_item_dim
                or any(int(target) not in positions for target in targets)):
            raise ValueError("invalid ContextBlock/target pairing")
        self._sync_stacked_to_models()
        obs = self._to_float_tensor(kwargs["obs_i_batch"], self.obs_dim)
        z = self._to_float_tensor(kwargs["z_core_excl_j_batch"], self.core_dim)
        m = self._to_float_tensor(kwargs["m_periph_excl_j_batch"], self.periph_dim)
        belief = self._to_float_tensor(kwargs["belief_summary_batch"], self.belief_dim)
        action_i = self._one_hot(np.asarray(kwargs["action_i_batch"], dtype=np.int64))
        pair = self._pair_feat_tensor(kwargs.get("pair_feat_batch"), len(targets))
        raw = torch.tensor(items, dtype=torch.float32, device=self.device)
        target_idx = torch.tensor([positions[int(t)] for t in targets], dtype=torch.long, device=self.device)
        outputs = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                encoded = model.context_item_encoder(raw)
                loo = encoded.sum(dim=0, keepdim=True) - encoded.index_select(0, target_idx)
                outputs.append(model(
                    obs, action_i, z, m, belief, pair_feat=pair,
                    context_embedding=loo,
                ))
        return self.score_batch_full(
            **kwargs, lag_preds_all_override=torch.stack(outputs, dim=0)
        )

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
            "latest_dr_applied": bool(self.latest_dr_applied),
            "latest_dr_applied_rows": int(self.latest_dr_applied_rows),
            "latest_dr_clipped_rows": int(self.latest_dr_clipped_rows),
            "latest_dr_raw_inverse_max": float(self.latest_dr_raw_inverse_max),
            "total_dr_applied_calls": int(self.total_dr_applied_calls),
            "total_dr_applied_rows": int(self.total_dr_applied_rows),
            "total_dr_clipped_rows": int(self.total_dr_clipped_rows),
            "effect_mode": self.effect_mode,
            "use_doubly_robust": bool(self.use_doubly_robust),
            "n_ensemble": int(self.n_ensemble),
            "n_horizons": int(self.n_horizons),
            "use_vmap_ensemble": bool(self.use_vmap_ensemble),
        }


# =========================================================================
# [FIX-X1] x_ij — pair features for Equation 8
# =========================================================================

PAIR_FEAT_DIM = 5 + len(PUBLIC_ROLES)


def build_pair_feat(
    positions,
    agent_zone,
    grid_size,
    n_zones,
    ego,
    j,
    agent_role=None,
):
    """Build ``x_ij`` from pre-treatment observable pair state.

    The channels are relative row, relative column, L1 distance, same-zone,
    normalized zone difference, and a one-hot target public role vector.
    OmniArena exposes role in every observed neighbour record; it is a task
    identity available before the intervention, not the oracle influence label.
    Environments without public roles receive a neutral zero role feature.

    WHY THIS IS REQUIRED (see FIX-X1 in LocalCounterfactualProxyNet.__init__):
    In omni_arena, w_ij(s)=phi_ij*delta_ij(s), where delta_ij(s) is purely a
    function of position: every _gate_ladder branch uses _dist(pos, anchor).
    Without x_ij, f_theta has no input distinguishing j and therefore cannot
    represent w_ij(s), regardless of training duration.

    Use this same function on both the replay_builder push path and
    final_runner scoring path. Divergent x_ij definitions cause silent,
    difficult-to-detect train/serve skew.
    """
    return OmniArenaAdapter._pair_features_from_tables(
        positions, agent_zone, grid_size, n_zones, ego, j, agent_role
    )
