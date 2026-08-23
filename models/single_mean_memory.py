"""Single-aggregate peripheral encoder for the H3 one-module ablation.

The class deliberately implements the same 11-dimensional item protocol as
``PeripheralMultiMemory`` and differs only at aggregation: all peripheral
items are compressed into one confidence-weighted mean.  Keeping the input
features, signature source, optimizer path, output width, and leave-one-out
contexts unchanged prevents the no-memory control from silently removing
forcing, doubly-robust estimation, or action-time feature storage as well.
"""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.causal_adapter import compact_relation_features, resolve_env_adapter

from models.influence_signature import SIGNATURE_DIM
from models.peripheral_memory import (
    FULL_ITEM_DIM,
    ITEM_ABS_MU,
    ITEM_ACTION,
    ITEM_CONTEXT_STD,
    ITEM_DIRECTION,
    ITEM_SIGMA_CAPACITY,
    LEGACY_ITEM_DIM,
)


class SingleMeanPeripheral(nn.Module):
    """Confidence-weighted single mean with the full H3 item interface."""

    is_single_mean = True

    def __init__(
        self,
        action_dim: int,
        memory_dim: int = 32,
        out_dim: int = 64,
        item_hidden: int = 48,
        *,
        signature_mode: str = "full",
        require_full_signature: bool = False,
        allow_legacy_items: bool = True,
        mu_floor: float = 0.02,
        beta_floor: float = 0.05,
        beta_mode: str = "capacity",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.memory_dim = int(memory_dim)
        self.out_dim = int(out_dim)
        self.item_hidden = int(item_hidden)
        self.item_dim = int(FULL_ITEM_DIM)
        self.signature_mode = str(signature_mode).strip().lower()
        if self.signature_mode not in {"full", "scalar"}:
            raise ValueError("signature_mode must be 'full' or 'scalar'")

        self.require_full_signature = bool(require_full_signature)
        self.allow_legacy_items = bool(allow_legacy_items)
        self.mu_floor = float(mu_floor)
        self.beta_floor = float(beta_floor)
        self.beta_mode = str(beta_mode).strip().lower()
        if self.beta_mode not in {"capacity", "abs_direction", "attention"}:
            raise ValueError("beta_mode must be capacity, abs_direction, or attention")
        self.eps = float(eps)

        self.signature_full_items_seen = 0
        self.signature_legacy_items_seen = 0
        self.last_signature_source = "none"

        self.encoder_in_dim = self.action_dim + self.item_dim - 1
        self.item_encoder = nn.Sequential(
            nn.Linear(self.encoder_in_dim, self.item_hidden),
            nn.ReLU(),
            nn.Linear(self.item_hidden, self.memory_dim),
            nn.LayerNorm(self.memory_dim),
        )
        self.importance_attention = nn.Sequential(
            nn.Linear(self.item_dim, self.item_hidden),
            nn.ReLU(),
            nn.Linear(self.item_hidden, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(self.memory_dim, self.out_dim),
            nn.ReLU(),
        )

        # FinalCIGAMFRunner's threshold calibration reads this field and calls
        # set_role_thresholds when available.  It is intentionally inert here:
        # a single mean has no semantic gates.
        self.register_buffer("g_anom_usage_ema", torch.zeros(1))

    def _device(self):
        return next(self.parameters()).device

    def reset_diagnostics(self):
        """Discard training-time input counters before held-out evaluation."""
        self.signature_full_items_seen = 0
        self.signature_legacy_items_seen = 0
        self.last_signature_source = "none"

    def set_role_thresholds(self, tau_role, sigma_hi, sigma_iqr=None):
        """Compatibility no-op; a single aggregate has no role thresholds."""
        del tau_role, sigma_hi, sigma_iqr

    def _normalise_inputs(self, periph_items) -> torch.Tensor:
        device = self._device()
        if periph_items is None:
            return torch.zeros(0, self.item_dim, dtype=torch.float32, device=device)
        if isinstance(periph_items, torch.Tensor):
            items = periph_items.to(device=device, dtype=torch.float32)
        else:
            items = torch.as_tensor(
                np.asarray(periph_items, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
        if items.ndim == 1:
            items = items.unsqueeze(0)
        if items.numel() == 0:
            return torch.zeros(0, self.item_dim, dtype=torch.float32, device=device)
        if items.shape[-1] == LEGACY_ITEM_DIM:
            if self.require_full_signature or not self.allow_legacy_items:
                raise ValueError(
                    "Received legacy 9D peripheral items, but this module "
                    "requires the tracker-derived 5D influence signature"
                )
            legacy = items
            upgraded = torch.zeros(
                legacy.shape[0], self.item_dim,
                dtype=torch.float32, device=device,
            )
            upgraded[:, 0] = legacy[:, 0]
            upgraded[:, 1] = legacy[:, 1]
            upgraded[:, 2] = torch.abs(legacy[:, 1])
            upgraded[:, 3] = legacy[:, 2]
            upgraded[:, 6] = 0.0
            upgraded[:, 7:] = legacy[:, 5:]
            items = upgraded
        if items.shape[-1] != self.item_dim:
            raise ValueError(
                f"SingleMeanPeripheral expected {self.item_dim}D full or "
                f"{LEGACY_ITEM_DIM}D legacy items, got {items.shape[-1]}"
            )
        return items

    def _prepare_encoder_input(self, items: torch.Tensor) -> torch.Tensor:
        actions = items[:, ITEM_ACTION].long().clamp(0, self.action_dim - 1)
        action_oh = F.one_hot(actions, num_classes=self.action_dim).float()
        rest = items[:, 1:].clone()
        if self.signature_mode == "scalar":
            rest[:, ITEM_ABS_MU - 1:ITEM_CONTEXT_STD] = 0.0
        return torch.cat([action_oh, rest], dim=-1)

    def _importance_beta(self, items: torch.Tensor) -> torch.Tensor:
        mu = items[:, 1]
        sigma = (
            torch.zeros_like(mu)
            if self.signature_mode == "scalar"
            else torch.clamp(items[:, ITEM_SIGMA_CAPACITY], min=0.0)
        )
        confidence = 1.0 / (1.0 + sigma + self.eps)
        if self.beta_mode == "capacity":
            beta = (torch.abs(mu) + self.mu_floor) * confidence
        elif self.beta_mode == "abs_direction":
            beta = (torch.abs(items[:, ITEM_DIRECTION]) + self.mu_floor) * confidence
        else:
            beta = torch.softmax(
                self.importance_attention(items).squeeze(-1), dim=0
            )
        return torch.clamp(beta, min=self.eps)

    def _encode_and_weight(self, items: torch.Tensor):
        encoded = self.item_encoder(self._prepare_encoder_input(items))
        weights = self._importance_beta(items)
        return encoded, weights

    def forward_full(self, periph_items) -> Dict[str, torch.Tensor]:
        items = self._normalise_inputs(periph_items)
        device = self._device()
        zero = torch.zeros((), dtype=torch.float32, device=device)
        if items.shape[0] == 0:
            pooled = torch.zeros(self.memory_dim, dtype=torch.float32, device=device)
        else:
            encoded, weights = self._encode_and_weight(items)
            pooled = torch.sum(weights[:, None] * encoded, dim=0) / torch.clamp(
                weights.sum(), min=self.eps
            )
        return {
            "memory": self.out_proj(pooled),
            "lb_loss": zero,
            "orth_loss": zero,
            "aux_loss": zero,
            "slot_probs": torch.ones(
                items.shape[0], 1, dtype=torch.float32, device=device
            ),
            "semantic_probs": torch.empty(
                items.shape[0], 0, dtype=torch.float32, device=device
            ),
            "balance_probs": None,
            "slot_usage": torch.ones(1, dtype=torch.float32, device=device),
            "memories": pooled.unsqueeze(0),
        }

    def forward(self, periph_items):
        return self.forward_full(periph_items)["memory"]

    def forward_excluding_all(self, periph_items, item_ids) -> Dict[int, torch.Tensor]:
        items = self._normalise_inputs(periph_items)
        n_items = int(items.shape[0])
        if n_items == 0:
            return {}
        if len(item_ids) != n_items:
            raise ValueError("item_ids must correspond one-to-one with input rows")

        with torch.no_grad():
            encoded, weights = self._encode_and_weight(items)
            numerator = torch.sum(weights[:, None] * encoded, dim=0)
            denominator = weights.sum()
            numerators = numerator[None, :] - weights[:, None] * encoded
            denominators = torch.clamp(denominator - weights, min=self.eps)
            pooled = numerators / denominators[:, None]
            # For a singleton set, excluding the only item must equal the
            # explicitly defined empty-set representation rather than 0/eps
            # contaminated by floating-point cancellation.
            if n_items == 1:
                pooled.zero_()
            outputs = self.out_proj(pooled)
        return {int(item_ids[index]): outputs[index] for index in range(n_items)}

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
        context_validity: Optional[Mapping[int, float]] = None,
        require_full_signature: Optional[bool] = None,
    ) -> np.ndarray:
        """Build the same full 11D rows used by the multi-slot encoder."""
        ego_id = int(ego_id)
        prev_core_set = set() if prev_core_set is None else set(prev_core_set)
        require_full = (
            self.require_full_signature
            if require_full_signature is None
            else bool(require_full_signature)
        )
        ids = [int(j) for j in peripheral_ids if int(j) != ego_id]
        if not ids:
            return np.zeros((0, self.item_dim), dtype=np.float32)

        adapter = resolve_env_adapter(env)
        last_actions = getattr(env, "last_actions", [0] * int(env.n_agents))
        rows = []
        full_count = 0
        legacy_count = 0
        for neighbor_id in ids:
            relation = compact_relation_features(
                adapter, ego_id, neighbor_id, width=4
            )
            belief = belief_state[neighbor_id]
            signature = None
            if influence_signatures is not None:
                try:
                    signature = np.asarray(
                        influence_signatures[neighbor_id], dtype=np.float32
                    ).reshape(-1)
                except (KeyError, TypeError, ValueError):
                    signature = None
            if signature is None:
                if require_full:
                    raise ValueError(
                        "Full 5D peripheral signature required but missing "
                        f"for ego={ego_id}, neighbour={neighbor_id}"
                    )
                mu = float(belief["mu_bar"])
                signature = np.asarray(
                    [abs(mu), mu, float(belief["sigma_bar"]), float(belief["sigma_bar"]), 0.0],
                    dtype=np.float32,
                )
                legacy_count += 1
            else:
                if signature.shape[0] != SIGNATURE_DIM:
                    raise ValueError(
                        f"Influence signature for ego={ego_id}, "
                        f"neighbour={neighbor_id} must have {SIGNATURE_DIM} "
                        f"values, got {signature.shape[0]}"
                    )
                if not np.all(np.isfinite(signature)):
                    raise ValueError("Influence signature contains non-finite values")
                full_count += 1

            rows.append([
                float(np.clip(last_actions[neighbor_id], 0, self.action_dim - 1)),
                *[float(value) for value in signature],
                float(
                    0.0
                    if context_validity is None
                    else context_validity.get(neighbor_id, 0.0)
                ),
                *[float(value) for value in relation],
            ])

        self.signature_full_items_seen += full_count
        self.signature_legacy_items_seen += legacy_count
        if full_count and legacy_count:
            self.last_signature_source = "mixed"
        elif full_count:
            self.last_signature_source = "full_5d"
        else:
            self.last_signature_source = "legacy_derived"
        return np.asarray(rows, dtype=np.float32)

    def get_input_diagnostics(self):
        total = self.signature_full_items_seen + self.signature_legacy_items_seen
        return {
            "signature_source": self.last_signature_source,
            "signature_full_fraction": (
                float(self.signature_full_items_seen) / float(total)
                if total else float("nan")
            ),
            "signature_full_items_seen": int(self.signature_full_items_seen),
            "signature_legacy_items_seen": int(self.signature_legacy_items_seen),
            "require_full_signature": bool(self.require_full_signature),
        }
