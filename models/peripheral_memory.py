"""
peripheral_memory_v2.py — Peripheral multi-memory with a slot-collapse fix.

=============================================================================
DIAGNOSIS OF THE V1 NULL RESULT
=============================================================================
The paper's result table reports:
    NoMultiMemory (WITHOUT multi-memory)  Core F1 = 0.251 / 0.259, throughput 241.8
    Final CIG-AMF (WITH multi-memory)      Core F1 = 0.244 / 0.262, throughput  74.7

Removing the main module produced equivalent performance while running 3.2
TIMES FASTER. The module therefore consumed approximately 70% of the
computation without providing a measurable benefit.

ROOT CAUSE — SLOT COLLAPSE. The v1 code was:

    slot_logits = self.slot_router(enc_in)      # no task assigned to any slot
    slot_probs  = F.softmax(slot_logits, dim=-1)

No training signal specified that one slot should contain one type of item and
another slot should contain a different type. The network was only told that
any partition was acceptable as long as reward improved. Reward is delayed
and noisy, so the gradient reaching the slot-assignment layer was extremely
weak.

Without an assigned task, the network selected the easiest solution. Two
collapse modes were possible:
  (a) UNIFORM COLLAPSE: softmax assigned every item to all four slots with
      approximately equal weights (about 25% per slot). All four slots then
      contained the same mixture, each slot approximated the global mean, and
      concatenating four slots produced four copies of a single mean. Four
      containers were present, but all contained the same representation.
  (b) MONOPOLY COLLAPSE: one slot absorbed everything and the other three
      remained empty, again reducing the representation to approximately one
      mean.

v1 also used `uniform_mix = 0.25` to mix a uniform memory into the slots. This
MADE mode (a) WORSE by actively pulling every slot toward the common mean.

=============================================================================
THREE CORRECTION LAYERS, USED TOGETHER
=============================================================================

[T1] ASSIGN TASKS TO SLOTS — semantic slots.
     Instead of leaving softmax unconstrained, each slot has a predefined
     functional role inferred from its causal influence signature:
        slot 0, "Beneficial": mu > 0, strong, and certain
        slot 1, "Harmful"   : mu < 0, strong, and certain
        slot 2, "Neutral"   : |mu| approximately 0
        slot 3, "Anomalous" : high sigma — not yet understood, representing
                               agents whose effects remain uncertain
     Assignment is soft through sigmoid gates, so gradients still flow.

     Compared with k-means, semantic slot meanings remain FIXED over time.
     K-means would require periodic reclustering, and cluster meanings could
     change after every reclustering. The policy would then have to relearn
     their interpretation, introducing another source of non-stationarity —
     exactly the phenomenon the paper is intended to address.

     The "Anomalous" slot is particularly important: it turns sigma from a
     control parameter into a SEMANTIC DIMENSION. Most methods treat
     uncertainty only as something to reduce; here it is an attribute used
     for classification.

[T2] PREVENT MONOPOLY COLLAPSE — load-balancing loss (Switch Transformer).
        L_lb = alpha * K * sum_q f_q * P_q
     f_q is the fraction of items routed to slot q.
     P_q is the mean routing probability assigned to slot q by the router.
     Both equal 1/K under perfect balance, where their product is minimized.
     The gradient scales with overload, creating a self-correcting feedback
     loop. Fedus et al. swept alpha from 1e-1 to 1e-5 and recommended 1e-2.

[T3] PREVENT UNIFORM-CONTENT COLLAPSE — orthogonality loss.
        L_orth = mean_{q != r} cosine_similarity(m_q, m_r)^2
     This penalizes slot vectors that are too similar. It directly prevents
     four slots from carrying the same content. Load balancing cannot correct
     mode (a), because routing can be perfectly even while slot contents remain
     identical; both losses are therefore necessary.

     An optional ROMA-style mutual-information regularizer is also available;
     see slot_specialisation_loss.
=============================================================================
"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.influence_signature import (
    N_SEMANTIC_ROLES,
    ROLE_ANOMALOUS,
    ROLE_BENEFICIAL,
    ROLE_HARMFUL,
    ROLE_NEUTRAL,
    SIGNATURE_DIM,
)


# Full H3 items contain the five-dimensional influence signature followed by
# belief and geometry fields.  The nine-dimensional layout used before H3 was
# wired to the tracker is still accepted, but is upgraded explicitly and is
# reported as ``legacy_derived`` in diagnostics.  A run that claims to test the
# full signature must set ``require_full_signature=True``.
LEGACY_ITEM_DIM = 9
FULL_ITEM_DIM = 12

ITEM_ACTION = 0
ITEM_CAPACITY = 1
ITEM_DIRECTION = 2
ITEM_SIGMA_CAPACITY = 3
ITEM_SIGMA_DIRECTION = 4
ITEM_CONTEXT_STD = 5
# Compatibility aliases for callers that still import the former names.
ITEM_SIGNED_MU = ITEM_DIRECTION
ITEM_ABS_MU = ITEM_CAPACITY
ITEM_SIGMA = ITEM_SIGMA_DIRECTION
ITEM_TEMPORAL_STD = ITEM_SIGMA_CAPACITY
ITEM_P_CORE = 6
ITEM_PREV_CORE = 7
ITEM_REL_ROW = 8
ITEM_REL_COL = 9
ITEM_ZONE_DIFF = 10
ITEM_DISTANCE = 11

ROUTING_MODES = ("semantic", "unconstrained")
SIGNATURE_MODES = ("full", "scalar")


class PeripheralMultiMemory(nn.Module):
    """
    Peripheral encoder with semantic and free slots.

    Hybrid architecture: four fixed beneficial/harmful/neutral/anomalous
    semantic slots plus n_free_slots learned by the router to capture residual
    structure. Total slots equal 4+n_free_slots.

    Full item format has twelve dimensions:
        0: action_j
        1: structural capacity C
        2: behavioural direction D
        3: sigma_C
        4: sigma_D
        5: context_std(v_C)
        6: p_core
        7: in_prev_core
        8: rel_row
        9: rel_col
       10: zone_diff
       11: distance_norm

    Nine-dimensional legacy items are upgraded only for compatibility.  They
    do not provide separate C/D uncertainty and cannot support the redesigned
    representation claim.

    Input:
        periph_items: np/tensor [N_p, 9]
    Output:
        forward()      -> [out_dim]
        forward_full() returns memory and auxiliary losses.

    Args:
        n_free_slots:
            Free slots beyond the four semantic slots; zero means semantic only.
        lb_coeff:
            Load-balancing coefficient; Fedus et al. recommend 1e-2.
        orth_coeff:
            Orthogonality-loss coefficient.
        role_sharpness:
            Sigmoid slope for soft assignment; larger approaches hard assignment.
        use_uniform_mix:
            v1 uniform-memory mixing pulled every slot toward a
            common mean and induced uniform collapse. v2 disables it by
            default; enable only for the before-correction ablation.
        routing_mode:
            ``semantic`` uses fixed influence-role gates plus optional free
            slots. ``unconstrained`` routes all slots with a learned softmax
            and is the faithful no-semantic ablation.
        signature_mode:
            ``full`` uses all five signature dimensions. ``scalar`` masks all
            influence-signature channels except signed_mu while retaining
            action, belief membership, and geometry as controls.
    """

    def __init__(
        self,
        action_dim: int,
        memory_dim: int = 32,
        out_dim: int = 64,
        item_hidden: int = 48,
        item_dim: int = FULL_ITEM_DIM,
        n_free_slots: int = 2,
        # Role thresholds should come from tracker.auto_calibrate().
        tau_role: float = 0.05,
        sigma_hi: float = 0.5,
        role_sharpness: float = 3.0,
        # Regularization coefficients.
        lb_coeff: float = 1e-2,
        orth_coeff: float = 1e-2,
        # Backward compatibility.
        num_slots: Optional[int] = None,
        use_uniform_mix: bool = False,
        uniform_mix: float = 0.25,
        routing_mode: str = "semantic",
        signature_mode: str = "full",
        require_full_signature: bool = True,
        allow_legacy_items: bool = False,
        mu_floor: float = 0.02,
        beta_floor: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.action_dim = int(action_dim)
        self.memory_dim = int(memory_dim)
        self.out_dim = int(out_dim)
        self.item_hidden = int(item_hidden)
        self.item_dim = int(item_dim)

        if self.item_dim != FULL_ITEM_DIM:
            raise ValueError(
                f"PeripheralMultiMemory uses the {FULL_ITEM_DIM}D full item "
                f"layout; got item_dim={self.item_dim}. Pass legacy 9D arrays "
                "to forward only when allow_legacy_items=True."
            )

        self.routing_mode = str(routing_mode).strip().lower()
        self.signature_mode = str(signature_mode).strip().lower()
        if self.routing_mode not in ROUTING_MODES:
            raise ValueError(
                f"routing_mode must be one of {ROUTING_MODES}, got "
                f"{routing_mode!r}"
            )
        if self.signature_mode not in SIGNATURE_MODES:
            raise ValueError(
                f"signature_mode must be one of {SIGNATURE_MODES}, got "
                f"{signature_mode!r}"
            )

        self.require_full_signature = bool(require_full_signature)
        self.allow_legacy_items = bool(allow_legacy_items)
        self.signature_full_items_seen = 0
        self.signature_legacy_items_seen = 0
        self.last_signature_source = "none"

        self.n_semantic_slots = int(N_SEMANTIC_ROLES)
        self.n_free_slots = int(max(0, n_free_slots))

        # Interpret legacy num_slots as the total slot count.
        if num_slots is not None:
            total = int(num_slots)
            if total < self.n_semantic_slots:
                raise ValueError(
                    f"num_slots={total} is smaller than the four fixed "
                    "semantic slots"
                )
            self.n_free_slots = int(max(0, total - self.n_semantic_slots))

        self.num_slots = self.n_semantic_slots + self.n_free_slots

        self.tau_role = float(tau_role)
        self.sigma_hi = float(sigma_hi)
        self.role_sharpness = float(role_sharpness)
        self.sigma_iqr_floor = float(sigma_hi)

        self.register_buffer("g_anom_usage_ema", torch.zeros(1))
        self.g_anom_ema_alpha = 0.05

        self.lb_coeff = float(lb_coeff)
        self.orth_coeff = float(orth_coeff)

        self.use_uniform_mix = bool(use_uniform_mix)
        self.uniform_mix = float(uniform_mix)
        self.mu_floor = float(mu_floor)
        self.beta_floor = float(beta_floor)
        self.eps = float(eps)

        # Action one-hot plus the remaining item fields.
        self.non_action_dim = self.item_dim - 1
        self.encoder_in_dim = self.action_dim + self.non_action_dim

        self.item_encoder = nn.Sequential(
            nn.Linear(self.encoder_in_dim, self.item_hidden),
            nn.ReLU(),
            nn.Linear(self.item_hidden, self.memory_dim),
            # A terminal ReLU forced every slot into the positive orthant.
            # Weighted means of such vectors had cosine approximately one, so
            # the orthogonality objective had almost no useful geometry.  A
            # centered output preserves signed directions for cosine-based
            # specialization while retaining a nonlinear hidden layer.
            nn.LayerNorm(self.memory_dim),
        )

        # ---------------------------------------------------------------
        # The router controls only free slots; rules assign semantic slots.
        #
        # Condition the router on semantic group. Synthetic four-role data
        # empirically validated this design:
        # (blocker / relay / consumer / inert):
        #     global signature k-means: 0.767 purity
        #     semantic grouping plus within-group k-means: 0.967 purity
        # Global normalization is dominated by strong blocker/relay roles and
        # compresses weak consumer/inert roles. Sign grouping first yielded
        # 1.000 purity for blocker/consumer separation in the harmful group.
        #
        # Concatenate sem_probs to router input for differentiable within-group
        # specialization instead of hard grouping followed by discrete k-means.
        # ---------------------------------------------------------------
        if self.n_free_slots > 0:
            self.router_in_dim = self.encoder_in_dim + self.n_semantic_slots

            self.slot_router = nn.Sequential(
                nn.Linear(self.router_in_dim, self.item_hidden),
                nn.ReLU(),
                nn.Linear(self.item_hidden, self.n_free_slots),
            )
        else:
            self.router_in_dim = self.encoder_in_dim
            self.slot_router = None

        # Faithful no-semantic ablation.  This router is constructed for every
        # variant so switching modes does not mutate the module or optimizer
        # after runner construction.  Its parameters are dormant in semantic
        # runs and receive no gradients there.
        self.unconstrained_router = nn.Sequential(
            nn.Linear(self.encoder_in_dim, self.item_hidden),
            nn.ReLU(),
            nn.Linear(self.item_hidden, self.num_slots),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(self.num_slots * self.memory_dim, self.out_dim),
            nn.ReLU(),
        )

        # Slot-usage diagnostics required to demonstrate collapse removal.
        self.register_buffer(
            "slot_usage_ema",
            torch.zeros(self.num_slots),
        )
        self.register_buffer("slot_hard_usage_ema", torch.zeros(self.num_slots))
        self.register_buffer(
            "slot_memory_ema", torch.zeros(self.num_slots, self.memory_dim)
        )
        self.register_buffer(
            "slot_signature_ema", torch.zeros(self.num_slots, SIGNATURE_DIM)
        )
        self.register_buffer(
            "slot_signature_support_ema", torch.zeros(self.num_slots)
        )
        self.register_buffer("signature_mean_ema", torch.zeros(SIGNATURE_DIM))
        self.register_buffer("signature_sq_mean_ema", torch.zeros(SIGNATURE_DIM))
        self.register_buffer(
            "slot_role_joint_ema",
            torch.zeros(self.num_slots, self.n_semantic_slots),
        )
        self.register_buffer("assignment_entropy_ema", torch.zeros(1))
        self.register_buffer("assignment_max_prob_ema", torch.zeros(1))
        self.register_buffer(
            "slot_diag_updates", torch.zeros((), dtype=torch.long)
        )
        self.usage_ema_alpha = 0.05
    # =====================================================================
    # Helper
    # =====================================================================

    def _device(self):
        return next(self.parameters()).device

    def _one_hot_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [N] long -> [N, action_dim]"""
        a = actions.long().clamp(min=0, max=self.action_dim - 1)
        return F.one_hot(a, num_classes=self.action_dim).to(dtype=torch.float32)

    def _normalise_inputs(self, periph_items) -> torch.Tensor:
        device = self._device()

        if periph_items is None:
            return torch.zeros(0, self.item_dim, dtype=torch.float32, device=device)

        if isinstance(periph_items, np.ndarray):
            x = torch.from_numpy(periph_items).to(device=device, dtype=torch.float32)
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
            return torch.zeros(0, self.item_dim, dtype=torch.float32, device=device)

        if x.shape[-1] == LEGACY_ITEM_DIM:
            if self.require_full_signature or not self.allow_legacy_items:
                raise ValueError(
                    "Received legacy 9D peripheral items, but this module "
                    "requires the tracker-derived 5D influence signature"
                )
            legacy = x
            upgraded = torch.zeros(
                legacy.shape[0], FULL_ITEM_DIM,
                dtype=torch.float32, device=device,
            )
            upgraded[:, ITEM_ACTION] = legacy[:, 0]
            upgraded[:, ITEM_SIGNED_MU] = legacy[:, 1]
            upgraded[:, ITEM_ABS_MU] = torch.abs(legacy[:, 1])
            upgraded[:, ITEM_SIGMA] = legacy[:, 2]
            # temporal_std and context_std are unavailable in v1.
            upgraded[:, ITEM_P_CORE] = legacy[:, 3]
            upgraded[:, ITEM_PREV_CORE] = legacy[:, 4]
            upgraded[:, ITEM_REL_ROW:] = legacy[:, 5:]
            x = upgraded
            self.signature_legacy_items_seen += int(legacy.shape[0])
            self.last_signature_source = "legacy_derived"

        if x.shape[-1] != FULL_ITEM_DIM:
            raise ValueError(
                f"PeripheralMultiMemory expected {FULL_ITEM_DIM}D full or "
                f"{LEGACY_ITEM_DIM}D legacy items, "
                f"got {x.shape[-1]}"
            )

        return x

    def _prepare_encoder_input(self, items: torch.Tensor) -> torch.Tensor:
        """Convert full items to the action-one-hot encoder representation."""
        action_col = items[:, ITEM_ACTION].long().clamp(
            min=0, max=self.action_dim - 1
        )
        action_oh = self._one_hot_actions(action_col)  # [N, action_dim]
        rest = items[:, 1:].to(dtype=torch.float32).clone()

        # p_core is a calibrated structural diagnostic, not a peripheral
        # semantic feature.  Letting it affect routing/pooling reintroduced a
        # hidden partition -> proxy feedback path through memory values.
        rest[:, ITEM_P_CORE - 1] = 0.0

        if self.signature_mode == "scalar":
            # Keep D only; remove C, both uncertainty channels, and v_ctx.
            # Indices are relative to ``rest`` (item columns 1..).
            rest[:, [
                ITEM_CAPACITY - 1,
                ITEM_SIGMA_CAPACITY - 1,
                ITEM_SIGMA_DIRECTION - 1,
                ITEM_CONTEXT_STD - 1,
            ]] = 0.0

        return torch.cat([action_oh, rest], dim=-1)

    def set_ablation_modes(
        self,
        *,
        routing_mode: Optional[str] = None,
        signature_mode: Optional[str] = None,
        require_full_signature: Optional[bool] = None,
    ):
        """Set H3 modes without rebuilding the runner or its optimizer."""
        if routing_mode is not None:
            mode = str(routing_mode).strip().lower()
            if mode not in ROUTING_MODES:
                raise ValueError(
                    f"routing_mode must be one of {ROUTING_MODES}, got {mode!r}"
                )
            self.routing_mode = mode
        if signature_mode is not None:
            mode = str(signature_mode).strip().lower()
            if mode not in SIGNATURE_MODES:
                raise ValueError(
                    f"signature_mode must be one of {SIGNATURE_MODES}, got {mode!r}"
                )
            self.signature_mode = mode
        if require_full_signature is not None:
            self.require_full_signature = bool(require_full_signature)

    # =====================================================================
    # [T1] Differentiable soft semantic-slot assignment.
    # =====================================================================

    def _semantic_slot_probs(self, items: torch.Tensor) -> torch.Tensor:        
        """
        Soft-assign each peripheral item to four influence-signature roles.

        items: [N,12], with the five signature fields in columns 1:6.

        Returns:
            [N,4], with each row summing to approximately one.

        Gate structure matching influence_signature.soft_role_assignment:
            g_anom = sigmoid(k*(sigma-sigma_hi))         unresolved
            g_sure = 1 - g_anom
            g_pos  = sigmoid(k*(mu-tau))                 beneficial
            g_neg  = sigmoid(k*(-mu-tau))                harmful
            g_neu  = clamp(1-g_pos-g_neg,0,1)            neutral

        Anomalous takes priority because assigning an uncertain item as
        beneficial or harmful would be arbitrary.
        """
        direction = items[:, ITEM_DIRECTION]
        if self.signature_mode == "scalar":
            # A genuine scalar-signature ablation has no uncertainty channel,
            # so it cannot use the anomalous role as a hidden second feature.
            sigma = torch.zeros_like(direction)
        else:
            sigma = torch.clamp(items[:, ITEM_SIGMA_DIRECTION], min=0.0)

        # Normalize slope by threshold. Otherwise at mu=0 sigmoid has not
        # saturated and neutral loses to beneficial/harmful. A unit test in
        # influence_signature.py exposed this defect.
        k_mu = self.role_sharpness / max(self.tau_role, 1e-8)
        # [B2.3] Scale g_anom by sigma-distribution dispersion rather than its
        # threshold. Threshold normalization is valid for mu because tau_role
        # is a |mu| percentile, but sigma_hi shrinks as beliefs converge while
        # the ensemble fix raised input sigma 16x. rho/sigma_hi then exploded,
        # saturated g_anom at one, routed every neighbour as anomalous, and
        # collapsed entropy. A floored IQR denominator stabilizes slope.
        k_sg_denom = max(
            float(getattr(self, "sigma_iqr_floor", self.sigma_hi)),
            float(self.sigma_hi) * 0.25,
            1e-3,
        )
        # [B2.3b] Cap the slope. Changing only the denominator reversed the
        # degeneracy: g_anom_mean was 0.0000 at ep15 and 0.0213 at ep50,
        # anomalous was almost unused, Hit Max Rate doubled 0.1429->0.2857,
        # and entropy fell 0.7949->0.7143. Both all-anomalous and none-anomalous
        # are degenerate. With sigma_hi at percentile 80, expected mean is
        # about 0.2. A finite cap of 12 keeps typical +/-0.05 differences at
        # sigmoid(+/-0.6)=0.35/0.65 and therefore retains a soft gate.
        k_sg = float(np.clip(self.role_sharpness / k_sg_denom, 1.0, 12.0))

        if self.signature_mode == "scalar":
            g_anom = torch.zeros_like(sigma)
        else:
            g_anom = torch.sigmoid(k_sg * (sigma - self.sigma_hi))
        g_sure = 1.0 - g_anom                                   # [N]

        g_pos = torch.sigmoid(k_mu * (direction - self.tau_role))      # [N]
        g_neg = torch.sigmoid(k_mu * (-direction - self.tau_role))     # [N]
        g_neu = torch.clamp(1.0 - g_pos - g_neg, min=0.0, max=1.0)  # [N]

        probs = torch.zeros(
            items.shape[0], self.n_semantic_slots,
            dtype=torch.float32, device=items.device,
        )  # [N, 4]

        probs[:, ROLE_BENEFICIAL] = g_sure * g_pos
        probs[:, ROLE_HARMFUL] = g_sure * g_neg
        probs[:, ROLE_NEUTRAL] = g_sure * g_neu
        probs[:, ROLE_ANOMALOUS] = g_anom

        row_sum = torch.clamp(probs.sum(dim=1, keepdim=True), min=self.eps)  # [N,1]

        with torch.no_grad():
            self.g_anom_usage_ema.mul_(1.0 - self.g_anom_ema_alpha).add_(
                self.g_anom_ema_alpha * g_anom.mean()
            )

        return probs / row_sum  # [N, 4]

    def _importance_beta(self, items: torch.Tensor) -> torch.Tensor:
        """
        Confidence weight for within-slot pooling.

        Preserve v1's prioritization of strong effects via |mu|, but apply the
        absolute value here after sign has already selected the slot, not in
        the estimator.

        beta = beta_floor * (C + mu_floor) * 1/(1+sigma_C)

        Returns: [N]
        """
        capacity = torch.clamp(items[:, ITEM_CAPACITY], min=0.0)
        sigma = (
            torch.zeros_like(capacity)
            if self.signature_mode == "scalar"
            else torch.clamp(items[:, ITEM_SIGMA_CAPACITY], min=0.0)
        )
        confidence = 1.0 / (1.0 + sigma + self.eps)  # [N]

        beta = (
            self.beta_floor
            * (capacity + self.mu_floor)
            * confidence
        )  # [N]

        return torch.clamp(beta, min=self.eps)

    def _route_items(
        self,
        items: torch.Tensor,
        enc_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return active, semantic-reference, and trainable-router probabilities."""
        sem_probs = self._semantic_slot_probs(items)

        if self.routing_mode == "unconstrained":
            logits = self.unconstrained_router(enc_in)
            probs = F.softmax(logits, dim=-1)
            return probs, sem_probs, probs

        if self.n_free_slots <= 0:
            return sem_probs, sem_probs, None

        # Free slots are conditioned on the fixed semantic group and learn
        # residual within-role structure.  Detaching the role posterior keeps
        # the semantic definition fixed while allowing the free router and
        # item encoder to learn end-to-end.
        router_in = torch.cat([enc_in, sem_probs.detach()], dim=-1)
        free_logits = self.slot_router(router_in)
        free_probs = F.softmax(free_logits, dim=-1)
        return (
            torch.cat([0.5 * sem_probs, 0.5 * free_probs], dim=1),
            sem_probs,
            free_probs,
        )

    def _update_slot_diagnostics(
        self,
        *,
        slot_probs: torch.Tensor,
        semantic_probs: torch.Tensor,
        memories: torch.Tensor,
        items: torch.Tensor,
    ):
        """Update streaming diagnostics that distinguish both collapse modes."""
        with torch.no_grad():
            n_items = int(slot_probs.shape[0])
            if n_items == 0:
                return

            usage = slot_probs.mean(dim=0)
            hard = F.one_hot(
                slot_probs.argmax(dim=1), num_classes=self.num_slots
            ).to(dtype=torch.float32)
            hard_usage = hard.mean(dim=0)

            p = torch.clamp(slot_probs, min=self.eps)
            conditional_entropy = -torch.mean(
                torch.sum(p * torch.log(p), dim=1)
            )
            max_prob = torch.mean(torch.max(slot_probs, dim=1).values)

            signatures = items[:, ITEM_CAPACITY:ITEM_CONTEXT_STD + 1]
            support = slot_probs.sum(dim=0)
            centroids = (slot_probs.t() @ signatures) / torch.clamp(
                support.unsqueeze(1), min=self.eps
            )

            semantic_hard = F.one_hot(
                semantic_probs.argmax(dim=1),
                num_classes=self.n_semantic_slots,
            ).to(dtype=torch.float32)
            role_joint = (slot_probs.t() @ semantic_hard) / float(n_items)

            first = int(self.slot_diag_updates.item()) == 0
            alpha = float(self.usage_ema_alpha)

            def update_ema(target, value):
                if first:
                    target.copy_(value)
                else:
                    target.mul_(1.0 - alpha).add_(alpha * value)

            update_ema(self.slot_usage_ema, usage)
            update_ema(self.slot_hard_usage_ema, hard_usage)
            update_ema(self.slot_memory_ema, memories.detach())
            update_ema(self.slot_signature_ema, centroids)
            update_ema(
                self.slot_signature_support_ema,
                support / float(n_items),
            )
            update_ema(self.signature_mean_ema, signatures.mean(dim=0))
            update_ema(
                self.signature_sq_mean_ema,
                torch.mean(signatures ** 2, dim=0),
            )
            update_ema(self.slot_role_joint_ema, role_joint)
            update_ema(
                self.assignment_entropy_ema,
                conditional_entropy.reshape_as(self.assignment_entropy_ema),
            )
            update_ema(
                self.assignment_max_prob_ema,
                max_prob.reshape_as(self.assignment_max_prob_ema),
            )
            self.slot_diag_updates.add_(1)

    # =====================================================================
    # [T2][T3] Auxiliary losses.
    # =====================================================================

    def _load_balancing_loss(self, slot_probs: torch.Tensor) -> torch.Tensor:
        """
        [T2] Switch Transformer load-balancing loss.

            L = K * sum_q  f_q * P_q

        slot_probs: [N, K] for an exchangeable trainable router. Fixed
        semantic-role probabilities must not be passed here because their
        empirical prevalence is not expected to be uniform.

        f_q is the discrete argmax routing fraction and acts as a coefficient;
        P_q is mean routing probability and carries gradients.

        Perfect balance gives f_q=P_q=1/K and L=1. Imbalance increases L.

        Returns: scalar
        """
        N, K = slot_probs.shape

        if N == 0:
            return torch.zeros((), dtype=torch.float32, device=slot_probs.device)

        # f_q is the mean argmax one-hot and needs no gradient.
        with torch.no_grad():
            hard = F.one_hot(
                slot_probs.argmax(dim=1), num_classes=K
            ).to(dtype=torch.float32)  # [N, K]
            f = hard.mean(dim=0)       # [K]

        P = slot_probs.mean(dim=0)     # [K], gradient path.

        return float(K) * torch.sum(f * P)

    def _orthogonality_loss(
        self,
        memories: torch.Tensor,
        slot_support: torch.Tensor,
    ) -> torch.Tensor:
        """
        [T3] Penalize overly similar slot vectors to prevent uniform content.

        memories: [K, memory_dim]

        L = mean over q != r of  cos(m_q, m_r)^2

        Squared cosine is nonnegative and penalizes both identical and
        antipodal vectors. Antipodal beneficial/harmful semantics may be valid,
        so allow_antipodal provides an alternative when opposites should not
        be penalized.

        Returns: scalar
        """
        K = memories.shape[0]

        if K < 2:
            return torch.zeros((), dtype=torch.float32, device=memories.device)

        supported = slot_support.reshape(-1) > self.eps
        if int(supported.sum().item()) < 2:
            return torch.zeros((), dtype=torch.float32, device=memories.device)

        normed = F.normalize(memories, p=2, dim=1, eps=self.eps)  # [K, D]
        gram = normed @ normed.t()                                 # [K, K]

        # Empty or near-empty slots have no semantic content.  Including them
        # lets the model lower this loss by killing slots instead of separating
        # their representations.
        mask = (
            ~torch.eye(K, dtype=torch.bool, device=memories.device)
            & supported.unsqueeze(0)
            & supported.unsqueeze(1)
        )
        off_diag = gram[mask]                                            # [K*(K-1)]

        if off_diag.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=memories.device)

        return torch.mean(off_diag ** 2)

    # =====================================================================
    # Forward
    # =====================================================================

    def forward_full(self, periph_items) -> Dict[str, torch.Tensor]:
        """
        Full forward pass returning memory and all auxiliary losses.

        Returns dict:
            memory:      [out_dim]           — supplied to the policy
            lb_loss:     scalar              — load balancing
            orth_loss:   scalar              — orthogonality
            aux_loss:    scalar              — lb_coeff*lb + orth_coeff*orth
            slot_probs:  [N, K]              — diagnostics
            slot_usage:  [K]                 — usage fraction per slot
            memories:    [K, memory_dim]     — vector for each slot
        """
        items = self._normalise_inputs(periph_items)  # [N, 9]
        device = self._device()

        zero = torch.zeros((), dtype=torch.float32, device=device)

        # Empty input.
        if items.shape[0] == 0:
            x = torch.zeros(
                1, self.num_slots * self.memory_dim,
                dtype=torch.float32, device=device,
            )
            return {
                "memory": self.out_proj(x).squeeze(0),   # [out_dim]
                "lb_loss": zero,
                "orth_loss": zero,
                "aux_loss": zero,
                "slot_probs": torch.zeros(
                    0, self.num_slots, dtype=torch.float32, device=device
                ),
                "semantic_probs": torch.zeros(
                    0, self.n_semantic_slots,
                    dtype=torch.float32, device=device,
                ),
                "balance_probs": None,
                "slot_usage": torch.zeros(
                    self.num_slots, dtype=torch.float32, device=device
                ),
                "memories": torch.zeros(
                    self.num_slots, self.memory_dim,
                    dtype=torch.float32, device=device,
                ),
            }

        N = items.shape[0]

        enc_in = self._prepare_encoder_input(items)   # [N, action_dim+8]
        h = self.item_encoder(enc_in)                 # [N, memory_dim]

        slot_probs, sem_probs, balance_probs = self._route_items(items, enc_in)

        # Pool within each slot.
        beta = self._importance_beta(items)           # [N]

        # weighted[n, q] = slot_probs[n,q] * beta[n]
        weighted = slot_probs * beta.unsqueeze(1)     # [N, K]

        # memories[q] = sum_n weighted[n,q] * h[n] / sum_n weighted[n,q]
        num = weighted.t() @ h                        # [K, memory_dim]
        den = torch.clamp(
            weighted.sum(dim=0), min=self.eps
        ).unsqueeze(1)                                # [K, 1]

        memories = num / den                          # [K, memory_dim]

        # v1 uniform mix, disabled by default. Enable only for ablation: it
        # actively pulls every slot toward the global mean and CAUSES uniform
        # collapse.
        if self.use_uniform_mix:
            num_u = slot_probs.t() @ h                              # [K, D]
            den_u = torch.clamp(
                slot_probs.sum(dim=0), min=self.eps
            ).unsqueeze(1)                                          # [K, 1]
            uniform_mem = num_u / den_u                             # [K, D]

            mix = float(np.clip(self.uniform_mix, 0.0, 1.0))
            memories = (1.0 - mix) * memories + mix * uniform_mem

        # Auxiliary losses.
        # Semantic roles are fixed and need not occur equally often. Applying
        # a Switch-style equal-load objective to them is both conceptually
        # wrong and gradient-free because their gates contain no trainable
        # parameters. Balance only the exchangeable learned router: free slots
        # in Full, or every slot in the unconstrained ablation.
        lb_loss = (
            self._load_balancing_loss(balance_probs)
            if balance_probs is not None
            else zero
        )
        orth_loss = self._orthogonality_loss(
            memories,
            slot_support=weighted.sum(dim=0),
        )
        aux_loss = self.lb_coeff * lb_loss + self.orth_coeff * orth_loss

        usage = slot_probs.mean(dim=0)
        self._update_slot_diagnostics(
            slot_probs=slot_probs,
            semantic_probs=sem_probs,
            memories=memories,
            items=items,
        )

        flat = memories.reshape(1, -1)                # [1, K*memory_dim]
        memory_out = self.out_proj(flat).squeeze(0)   # [out_dim]

        return {
            "memory": memory_out,
            "lb_loss": lb_loss,
            "orth_loss": orth_loss,
            "aux_loss": aux_loss,
            "slot_probs": slot_probs,
            "semantic_probs": sem_probs,
            "balance_probs": balance_probs,
            "slot_usage": usage,
            "memories": memories,
        }

    def forward_excluding_all(self, periph_items, item_ids) -> Dict[int, torch.Tensor]:
        """
        [GPU_OPTIMIZATION_CONTRACT.md section 2.1] Compute M_i^{-j}
        simultaneously for every j in one ego's current peripheral set using
        sum-minus-one. The old path called forward_full separately for each
        exclusion: build_inputs plus a full forward N times per ego, rerunning
        item_encoder/slot_router on almost the entire set each time.

        This is VALID ONLY FOR WEIGHTED-SUM POOLING. Eq. 25 is a weighted mean,
        permutation-invariant in the Deep Sets style. Each item contributes
        independently through h[n], slot_probs[n], and beta[n], with no
        cross-item normalization before pooling. item_encoder, semantic gates,
        and the free-slot router were verified to be per-item MLPs without
        cross-item BatchNorm or attention. If pooling later changes to
        attention or max, including the paper's proposed Set Transformer
        variant, this method is wrong and must revert to a separate
        forward_full call for every exclusion.

        Do not use this method for training because it does not compute
        lb_loss/orth_loss. It only constructs M_i^{-j} proxy context. Train the
        peripheral module with forward_full() on the complete set.

        Args:
            periph_items: [N,item_dim], the complete current peripheral set.
            item_ids: N neighbour IDs corresponding to the rows.

        Returns:
            {item_id: memory_out [out_dim]} for every item ID.
        """
        items = self._normalise_inputs(periph_items)
        N = items.shape[0]

        if N == 0:
            return {}

        with torch.no_grad():
            enc_in = self._prepare_encoder_input(items)   # [N, enc_in_dim]
            h = self.item_encoder(enc_in)                 # [N, D]
            slot_probs, _, _ = self._route_items(items, enc_in)

            beta = self._importance_beta(items)          # [N]
            weighted = slot_probs * beta.unsqueeze(1)     # [N, K]

            num = weighted.t() @ h                         # [K,D], complete sum.
            den = weighted.sum(dim=0)                       # [K]

            # Each item's per-slot contribution is [N,K,D], vectorized over N
            # instead of a Python loop.
            contrib = weighted.unsqueeze(2) * h.unsqueeze(1)   # [N, K, D]
            num_excl = num.unsqueeze(0) - contrib               # [N, K, D]
            den_excl = torch.clamp(
                den.unsqueeze(0) - weighted, min=self.eps
            ).unsqueeze(2)                                       # [N, K, 1]
            memories = num_excl / den_excl                        # [N, K, D]

            if self.use_uniform_mix:
                num_u = slot_probs.t() @ h                          # [K, D]
                den_u = slot_probs.sum(dim=0)                        # [K]
                contrib_u = slot_probs.unsqueeze(2) * h.unsqueeze(1)  # [N,K,D]
                num_u_excl = num_u.unsqueeze(0) - contrib_u
                den_u_excl = torch.clamp(
                    den_u.unsqueeze(0) - slot_probs, min=self.eps
                ).unsqueeze(2)
                uniform_mem = num_u_excl / den_u_excl                  # [N,K,D]

                mix = float(np.clip(self.uniform_mix, 0.0, 1.0))
                memories = (1.0 - mix) * memories + mix * uniform_mem

            flat = memories.reshape(N, -1)          # [N, K*memory_dim]
            outs = self.out_proj(flat)              # [N, out_dim]

        return {int(item_ids[n]): outs[n] for n in range(N)}

    def forward(self, periph_items) -> torch.Tensor:
        """
        Preserve the v1 signature and exact [out_dim] return. Legacy runners
        work unchanged; call forward_full() when auxiliary loss is required.
        """
        return self.forward_full(periph_items)["memory"]

    # =====================================================================
    # Input construction and explicit legacy compatibility.
    # =====================================================================

    def build_inputs(
        self,
        ego_id,
        peripheral_ids,
        env,
        belief_state,
        prev_core_set=None,
        influence_signatures: Optional[
            Mapping[int, Sequence[float]]
        ] = None,
        require_full_signature: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Build the item matrix for one ego agent.

        ``influence_signatures`` must map neighbour ID to the tracker output
        ``[C, D, sigma_C, sigma_D, v_ctx]``.  When omitted, an explicitly
        labelled compatibility vector is derived from the legacy belief.
        Compatibility
        data are rejected when ``require_full_signature`` is true.

        This distinction is an experiment-validity requirement: derived
        vectors cannot support a five-dimensional-versus-scalar H3 claim.

        Returns:
            np.ndarray float32 [len(peripheral_ids), 12]
        """
        ego_id = int(ego_id)
        prev_core_set = set() if prev_core_set is None else set(prev_core_set)
        require_full = (
            self.require_full_signature
            if require_full_signature is None
            else bool(require_full_signature)
        )

        ids = [int(j) for j in list(peripheral_ids) if int(j) != ego_id]

        if len(ids) == 0:
            return np.zeros((0, self.item_dim), dtype=np.float32)

        pi = env.positions[ego_id]
        grid_den = max(1, int(env.grid_size))
        zone_den = max(1, int(env.n_zones) - 1)

        last_actions = getattr(
            env, "last_actions", [0] * int(env.n_agents)
        )

        rows = []
        full_count = 0
        legacy_count = 0

        for j in ids:
            pj = env.positions[j]
            b = belief_state[j]

            action_j = int(np.clip(int(last_actions[j]), 0, self.action_dim - 1))

            signature = None
            if influence_signatures is not None:
                try:
                    signature = np.asarray(
                        influence_signatures[int(j)], dtype=np.float32
                    ).reshape(-1)
                except (KeyError, TypeError, ValueError):
                    signature = None

            if signature is None:
                if require_full:
                    raise ValueError(
                        "Full 5D peripheral signature required but missing "
                        f"for ego={ego_id}, neighbour={j}"
                    )
                mu_legacy = float(b["mu_bar"])
                sigma_legacy = float(b["sigma_bar"])
                signature = np.asarray(
                    [
                        abs(mu_legacy),
                        mu_legacy,
                        sigma_legacy,
                        sigma_legacy,
                        0.0,
                    ],
                    dtype=np.float32,
                )
                legacy_count += 1
            else:
                if signature.shape[0] != SIGNATURE_DIM:
                    raise ValueError(
                        f"Influence signature for ego={ego_id}, neighbour={j} "
                        f"must have {SIGNATURE_DIM} values, got "
                        f"{signature.shape[0]}"
                    )
                if not np.all(np.isfinite(signature)):
                    raise ValueError(
                        f"Influence signature for ego={ego_id}, neighbour={j} "
                        "contains non-finite values"
                    )
                full_count += 1

            rows.append([
                float(action_j),
                *[float(v) for v in signature],
                float(b["p_core"]),
                float(j in prev_core_set),
                float((pj[0] - pi[0]) / grid_den),
                float((pj[1] - pi[1]) / grid_den),
                float((env.agent_zone[j] - env.agent_zone[ego_id]) / zone_den),
                float(abs(pj[0] - pi[0]) + abs(pj[1] - pi[1])) / grid_den,
            ])

        self.signature_full_items_seen += int(full_count)
        self.signature_legacy_items_seen += int(legacy_count)
        if full_count and legacy_count:
            self.last_signature_source = "mixed"
        elif full_count:
            self.last_signature_source = "full_5d"
        else:
            self.last_signature_source = "legacy_derived"

        return np.asarray(rows, dtype=np.float32)

    # =====================================================================
    # Diagnostics.
    # =====================================================================

    def reset_slot_diagnostics(self):
        """Reset streaming statistics before a fixed held-out probe pass.

        Training-time calls are not exchangeable across variants: auxiliary
        objectives and routing change how often particular paths execute, and
        an exponential moving average over that stream overweights the most
        recent minibatches.  H3 therefore clears these buffers after training
        and recomputes diagnostics on the same ordered held-out states for
        every arm.
        """
        with torch.no_grad():
            self.slot_usage_ema.zero_()
            self.slot_hard_usage_ema.zero_()
            self.slot_memory_ema.zero_()
            self.slot_signature_ema.zero_()
            self.slot_signature_support_ema.zero_()
            self.signature_mean_ema.zero_()
            self.signature_sq_mean_ema.zero_()
            self.slot_role_joint_ema.zero_()
            self.assignment_entropy_ema.zero_()
            self.assignment_max_prob_ema.zero_()
            self.slot_diag_updates.zero_()
            self.g_anom_usage_ema.zero_()
        self.signature_full_items_seen = 0
        self.signature_legacy_items_seen = 0
        self.last_signature_source = "none"

    def get_slot_diagnostics(self) -> Dict[str, object]:
        """
        Anti-collapse evidence that must be included in the paper.

        usage_entropy_ratio:
            Usage-distribution entropy divided by log(K). Approximately 1.0
            means even use of all K slots; approximately 0.0 means monopoly
            collapse. High entropy alone does not establish absence of
            collapse: uniform-content collapse has perfect entropy 1.0 while
            remaining useless. Report orthogonality too, using a centroid
            heatmap.

        max_usage / min_usage:
            min_usage near zero indicates a dead slot.
        """
        usage = self.slot_usage_ema.detach().cpu().numpy()
        hard_usage = self.slot_hard_usage_ema.detach().cpu().numpy()
        K = int(usage.shape[0])
        n_updates = int(self.slot_diag_updates.item())

        def entropy_ratio(values):
            values = np.asarray(values, dtype=np.float64)
            total = float(values.sum())
            if total <= 1e-12 or K <= 1:
                return float("nan")
            probs = np.clip(values / total, 1e-12, 1.0)
            return float(-np.sum(probs * np.log(probs)) / np.log(K))

        max_entropy = float(np.log(K)) if K > 1 else 1.0
        usage_entropy_ratio = entropy_ratio(usage)
        hard_usage_entropy_ratio = entropy_ratio(hard_usage)
        p = np.clip(usage / max(float(usage.sum()), 1e-12), 1e-12, 1.0)
        marginal_entropy = float(-np.sum(p * np.log(p)))
        conditional_entropy = float(self.assignment_entropy_ema.item())
        assignment_mi_ratio = float(
            max(0.0, marginal_entropy - conditional_entropy)
            / max(max_entropy, 1e-12)
        )

        full_seen = int(self.signature_full_items_seen)
        legacy_seen = int(self.signature_legacy_items_seen)
        total_seen = full_seen + legacy_seen

        out = {
            "n_slots": K,
            "n_semantic_slots": int(self.n_semantic_slots),
            "n_free_slots": int(self.n_free_slots),
            "routing_mode": str(self.routing_mode),
            "signature_mode": str(self.signature_mode),
            "signature_source": str(self.last_signature_source),
            "signature_full_fraction": (
                float(full_seen) / float(total_seen) if total_seen else float("nan")
            ),
            "signature_full_items_seen": full_seen,
            "signature_legacy_items_seen": legacy_seen,
            "require_full_signature": bool(self.require_full_signature),
            "uniform_mix_enabled": bool(self.use_uniform_mix),
            "uniform_mix": float(self.uniform_mix),
            "diagnostic_updates": n_updates,
            "usage_entropy_ratio": usage_entropy_ratio,
            "hard_usage_entropy_ratio": hard_usage_entropy_ratio,
            "assignment_entropy_ratio": float(
                conditional_entropy / max(max_entropy, 1e-12)
            ),
            "assignment_mutual_info_ratio": assignment_mi_ratio,
            "mean_assignment_max_prob": float(
                self.assignment_max_prob_ema.item()
            ),
            "effective_soft_slots": float(np.exp(marginal_entropy)),
            "max_usage": float(np.max(usage)) if usage.size else float("nan"),
            "min_usage": float(np.min(usage)) if usage.size else float("nan"),
            "max_hard_usage": (
                float(np.max(hard_usage)) if hard_usage.size else float("nan")
            ),
            "min_hard_usage": (
                float(np.min(hard_usage)) if hard_usage.size else float("nan")
            ),
            "lb_coeff": float(self.lb_coeff),
            "orth_coeff": float(self.orth_coeff),
        }

        for q in range(min(K, self.n_semantic_slots)):
            name = ("beneficial", "harmful", "neutral", "anomalous")[q]
            out[f"usage_{name}"] = float(usage[q])

        support = self.slot_signature_support_ema.detach().cpu().numpy()
        mem = self.slot_memory_ema.detach().cpu().numpy()
        mem_norm = np.linalg.norm(mem, axis=1)
        active = np.flatnonzero((support > 1e-4) & (mem_norm > 1e-8))
        if active.size >= 2:
            active_mem = mem[active]
            normed = active_mem / np.clip(
                np.linalg.norm(active_mem, axis=1, keepdims=True), 1e-8, None
            )
            gram = normed @ normed.T
            off_mask = ~np.eye(active.size, dtype=bool)
            out["mean_offdiag_cosine"] = float(np.mean(np.abs(gram[off_mask])))
        else:
            out["mean_offdiag_cosine"] = float("nan")

        # Content separation in the actual five signature dimensions.  Scale
        # by the streaming item standard deviation so high-variance channels
        # do not dominate the distance.
        centroids = self.slot_signature_ema.detach().cpu().numpy()
        sig_mean = self.signature_mean_ema.detach().cpu().numpy()
        sig_sq = self.signature_sq_mean_ema.detach().cpu().numpy()
        sig_scale = np.sqrt(np.clip(sig_sq - sig_mean ** 2, 1e-8, None))
        active_sig = np.flatnonzero(support > 1e-4)
        if active_sig.size >= 2:
            standardized = centroids[active_sig] / sig_scale[None, :]
            diffs = standardized[:, None, :] - standardized[None, :, :]
            distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
            off = distances[~np.eye(active_sig.size, dtype=bool)]
            out["mean_signature_centroid_distance"] = float(np.mean(off))
            out["min_signature_centroid_distance"] = float(np.min(off))
        else:
            out["mean_signature_centroid_distance"] = float("nan")
            out["min_signature_centroid_distance"] = float("nan")

        # Normalized mutual information between active slots and the semantic
        # role posterior.  Unlike entropy alone, it is zero when every item is
        # diffusely copied into every slot.
        joint = self.slot_role_joint_ema.detach().cpu().numpy().astype(np.float64)
        joint_total = float(joint.sum())
        if joint_total > 1e-12:
            joint /= joint_total
            q = joint.sum(axis=1, keepdims=True)
            r = joint.sum(axis=0, keepdims=True)
            denom = np.clip(q @ r, 1e-12, None)
            nz = joint > 1e-12
            mi = float(np.sum(joint[nz] * np.log(joint[nz] / denom[nz])))
            hq = float(-np.sum(q[q > 1e-12] * np.log(q[q > 1e-12])))
            hr = float(-np.sum(r[r > 1e-12] * np.log(r[r > 1e-12])))
            out["semantic_role_nmi"] = float(
                mi / max(np.sqrt(max(hq, 0.0) * max(hr, 0.0)), 1e-12)
            )
        else:
            out["semantic_role_nmi"] = float("nan")

        mem_cos = float(out["mean_offdiag_cosine"])
        monopoly = bool(
            np.isfinite(hard_usage_entropy_ratio)
            and (
                hard_usage_entropy_ratio < 0.5
                or float(np.max(hard_usage)) > 0.90
            )
        )
        diffuse = bool(
            out["assignment_entropy_ratio"] > 0.90
            and assignment_mi_ratio < 0.10
        )
        uniform_content = bool(np.isfinite(mem_cos) and mem_cos > 0.95)
        out["monopoly_collapse"] = monopoly
        out["diffuse_assignment_collapse"] = diffuse
        out["uniform_content_collapse"] = uniform_content
        out["collapse_detected"] = bool(
            monopoly or diffuse or uniform_content
        )
        out["g_anom_mean"] = float(self.g_anom_usage_ema.item())
        return out

    def set_role_thresholds(self, tau_role: float, sigma_hi: float, sigma_iqr: float = None):
        """
        Update thresholds after tracker.auto_calibrate().

        IMPORTANT: mu scale depends entirely on environment reward scale. A
        hard-coded tau_role can make every neighbour neutral under small
        rewards or no neighbour neutral under large rewards. Both outcomes
        make semantic slots useless.
        """
        self.tau_role = float(tau_role)
        self.sigma_hi = float(sigma_hi)
        self.sigma_iqr_floor = float(sigma_iqr) if sigma_iqr is not None else float(sigma_hi)


# =========================================================================
# Optional ROMA-style MI regularizer as an orthogonality alternative.
# =========================================================================

def slot_specialisation_loss(
    slot_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    ROMA-style mutual-information specialization regularizer.

        I(slot; item) = H(E_n[p(slot|n)]) - E_n[H(p(slot|n))]

    H(mean) is entropy of aggregate usage. High values evenly use all slots and
    prevent monopoly collapse. E[H] is the mean entropy of individual
    assignments. Low values assign each item decisively rather than diffusely
    and prevent uniform-content collapse.

    Maximizing I is equivalent to minimizing -I; this function returns -I as loss.

    ROMA (Wang et al., ICML 2020) uses MI to bind role and trajectory. This
    method binds slot and influence signature. Its interventional signal differs
    from ROMA's observational signal, making this an adapted use rather than a
    direct copy.

    Args:
        slot_probs: [N, K]

    Returns:
        Scalar -I; lower values indicate stronger specialization.
    """
    if slot_probs.shape[0] == 0:
        return torch.zeros((), dtype=torch.float32, device=slot_probs.device)

    p = torch.clamp(slot_probs, min=eps)             # [N, K]

    marginal = p.mean(dim=0)                          # [K]
    h_marginal = -torch.sum(marginal * torch.log(marginal + eps))

    h_conditional = -torch.mean(torch.sum(p * torch.log(p), dim=1))

    mutual_info = h_marginal - h_conditional

    return -mutual_info


# Backward-compatible alias.
PeripheralMultiMemory = PeripheralMultiMemory
