"""
ego_conditioned_latent.py — correction for pair-specific relational z_ij.

V1 DEFECT

The paper claims that z_ij captures how neighbour j's behaviour relates
specifically to ego i's outcome rather than representing j through one global
latent. This is the second of the paper's four stated contributions. However,
Eq. 16 as implemented in core_behavior.py::train_bc was only:

    L_z = -log p(a_j^{t+1} | z_ij^t)
    # code: F.cross_entropy(self.bc_head(z_next), target_a_j)

Predicting a_j requires no information about i. Although x_full contains o_i
and a_i, gradients encourage the encoder to ignore them because they do not
reduce this loss. z_ij therefore converges to the same global opponent model
for every ego. Pair specificity was invalidated by the loss itself and the
paper's second contribution had no enforcing mechanism.

The direct diagnostic is cosine similarity between z_ij and z_i'j for i!=i'
with the same j. A value near 1.0 means the latent is a global opponent model,
not pair-specific. pair_specificity_score() at the end of this file implements
that test; the pre-correction result should be retained as paper evidence.

THREE LOSS TERMS

[E1] C/D HEAD — predict [C_ij,D_ij] from z_ij. This directly requires z_ij to
     encode both whether j is consequential for i and how its current policy
     direction affects i. Unlike a_j, this target depends on both parties.

[E2] CONTRASTIVE EGO — separate z_ij from z_i'j. The same neighbour viewed by
     different egos should have different latents unless its effects are truly
     identical. InfoNCE positives are the same pair at different timesteps;
     negatives are the same j under different egos. ACORM contrasts agents;
     this method contrasts egos for the same agent, adapting the source idea
     along a different contrastive dimension.

[E3] Retain original L_z action prediction as an auxiliary term. It remains
     useful for tracking behavioural drift but is insufficient by itself.

    L_total = L_bc + w_inf * L_influence + w_con * L_contrastive
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EgoConditionedHeads(nn.Module):
    """
    Auxiliary prediction heads attached to the existing z_ij.

    They extend PairRelationalModule without rewriting it. During training,
    construct the heads and optimizer, then compute loss from z_batch,
    ego_ids, neighbour_ids, and C/D targets before the normal backward/update.

    latent_dim equals pair_rel_module.hidden_dim.  The auxiliary head predicts
    the structural-capacity/directional pair ``[C,D]``.  Latency remains a
    separate gated research path and is deliberately not overloaded onto this
    representation head.  proj_dim sets contrastive projection size.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 64,
        proj_dim: int = 32,
        temperature: float = 0.2,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.temperature = float(temperature)

        # E1: z_ij -> [C_ij, D_ij].
        self.cd_head = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), 2),
        )

        # E2: z_ij -> contrastive space
        self.proj_head = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(proj_dim)),
        )

    # =====================================================================
    # [E1] Influence prediction
    # =====================================================================

    def influence_loss(
        self,
        z: torch.Tensor,            # [B, latent_dim]
        cd_target: torch.Tensor,    # [B, 2]
    ) -> torch.Tensor:
        """
        Force z_ij to predict structural capacity C and behavioural direction D.

        Because w_ij depends on both i and j, successful prediction requires
        the encoder to use o_i and a_i, precisely the inputs the old a_j-only
        objective allowed it to ignore. Returns a scalar loss.
        """
        if z.shape[0] == 0:
            return torch.zeros((), dtype=torch.float32, device=z.device)

        pred = self.cd_head(z)  # [B, 2]
        target = cd_target
        if target.dim() != 2 or target.shape[-1] != 2:
            raise ValueError(
                "cd_target must have shape [batch, 2] containing [C, D]"
            )

        # Huber is more stable than MSE with outliers (proxy sometimes jumps).
        return F.smooth_l1_loss(pred, target)

    # =====================================================================
    # [E2] Contrastive ego
    # =====================================================================

    def contrastive_loss(
        self,
        z: torch.Tensor,               # [B, latent_dim]
        ego_ids: torch.Tensor,         # [B] long
        neighbor_ids: torch.Tensor,    # [B] long
        cd_targets: Optional[torch.Tensor] = None,  # [B, 2]
        profile_distance_threshold: float = 0.05,
        profile_temperature: float = 0.05,
    ) -> torch.Tensor:
        """
        InfoNCE with signal-aware same-neighbour/different-ego hard negatives.

        An anchor's positive is another sample from the same (ego,neighbour)
        pair and its negatives share neighbour j but have different egos. Do
        not use every other sample as a negative: z_ij and z_ik already differ
        naturally. The required constraint is that the same neighbour under
        different egos has different representations, because j may be a
        blocker for i and a relay for i'. Returns a scalar loss.
        """
        B = z.shape[0]

        if B < 3:
            return torch.zeros((), dtype=torch.float32, device=z.device)

        p = F.normalize(self.proj_head(z), p=2, dim=1, eps=1e-8)  # [B, proj_dim]
        sim = (p @ p.t()) / self.temperature                       # [B, B]

        same_ego = ego_ids.view(-1, 1) == ego_ids.view(1, -1)      # [B, B]
        same_nb = neighbor_ids.view(-1, 1) == neighbor_ids.view(1, -1)  # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=z.device)      # [B, B]

        pos_mask = same_ego & same_nb & (~eye)     # Same pair, different time point
        neg_mask = (~same_ego) & same_nb           # Same j, different ego
        neg_weight = neg_mask.to(dtype=z.dtype)
        if cd_targets is not None:
            if cd_targets.dim() != 2 or cd_targets.shape != (B, 2):
                raise ValueError("cd_targets must have shape [batch, 2]")
            profile_distance = torch.linalg.vector_norm(
                cd_targets.unsqueeze(1) - cd_targets.unsqueeze(0), dim=-1
            )
            # Same-ID samples are positives only within a stable causal
            # profile. Same-neighbour/different-ego negatives receive graded
            # weight according to how different their causal profiles are.
            pos_mask = pos_mask & (
                profile_distance <= float(profile_distance_threshold)
            )
            neg_weight = neg_mask.to(dtype=z.dtype) * torch.sigmoid(
                (profile_distance - float(profile_distance_threshold))
                / max(float(profile_temperature), 1e-6)
            )

        # Only for anchors with both positive and negative labels.
        valid = pos_mask.any(dim=1) & (neg_weight > 1e-8).any(dim=1)  # [B]

        if not bool(valid.any()):
            return torch.zeros((), dtype=torch.float32, device=z.device)

        neg_inf = torch.finfo(sim.dtype).min

        pos_sim = sim.masked_fill(~pos_mask, neg_inf)  # [B, B]
        cand_mask = pos_mask | (neg_weight > 1e-8)      # [B, B]
        log_weight = torch.zeros_like(sim)
        log_weight = torch.where(
            neg_mask,
            torch.log(torch.clamp(neg_weight, min=1e-8)),
            log_weight,
        )
        all_sim = (sim + log_weight).masked_fill(~cand_mask, neg_inf)

        # InfoNCE: -log( sum exp(pos) / sum exp(pos + neg) )
        log_num = torch.logsumexp(pos_sim[valid], dim=1)   # [n_valid]
        log_den = torch.logsumexp(all_sim[valid], dim=1)   # [n_valid]

        return torch.mean(log_den - log_num)

    # =====================================================================
    # Combined
    # =====================================================================

    def compute_loss(
        self,
        z: torch.Tensor,                       # [B, latent_dim]
        ego_ids: torch.Tensor,                 # [B]
        neighbor_ids: torch.Tensor,            # [B]
        cd_target: Optional[torch.Tensor] = None,  # [B,2]
        w_influence: float = 1.0,
        w_contrastive: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict:
            total       : scalar added to external L_bc
            influence   : scalar
            contrastive : scalar
        """
        device = z.device
        zero = torch.zeros((), dtype=torch.float32, device=device)

        l_inf = (
            self.influence_loss(z, cd_target) if cd_target is not None else zero
        )
        l_con = self.contrastive_loss(
            z, ego_ids, neighbor_ids, cd_targets=cd_target
        )

        total = w_influence * l_inf + w_contrastive * l_con

        return {"total": total, "influence": l_inf, "contrastive": l_con}


# =========================================================================
# Diagnostic: is the latent genuinely pair-specific?
# =========================================================================

def pair_specificity_score(
    pair_rel_module,
    n_agents: int,
    sample_neighbors: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Measure whether z_ij is genuinely pair-specific.

    Run this function before and after the correction. The paired measurements
    directly demonstrate whether the paper's second contribution has actually
    been implemented; the paper currently has no other supporting measurement.

    Method:
        For each neighbour j, collect {z_ij for every ego i != j} and compute
        mean cosine similarity across egos.

        ~1.0 means every ego represents j identically, so z is a global
        opponent model rather than pair-specific. A low value means egos
        represent j differently and supports the claimed ego-centric property.

    Compare against similarity among different neighbours under the same ego.
    If cross_ego_similarity is approximately cross_neighbor_similarity, the
    latent does not distinguish either dimension.

    Returns:
        Dictionary with cross_ego_similarity, cross_neighbor_similarity, and ratio.
    """
    ids = (
        list(range(int(n_agents)))
        if sample_neighbors is None
        else [int(x) for x in sample_neighbors]
    )

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))

        if na < 1e-8 or nb < 1e-8:
            return 0.0

        return float(np.dot(a, b) / (na * nb))

    # Same neighbour j under different egos.
    cross_ego = []

    for j in ids:
        vecs = []

        for i in range(int(n_agents)):
            if i == j:
                continue

            v = pair_rel_module.get_pair_latent(i, j)

            if v is not None and np.linalg.norm(v) > 1e-8:
                vecs.append(np.asarray(v, dtype=np.float64).reshape(-1))

        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                cross_ego.append(_cos(vecs[a], vecs[b]))

    # Same ego i with different neighbours.
    cross_nb = []

    for i in ids:
        vecs = []

        for j in range(int(n_agents)):
            if i == j:
                continue

            v = pair_rel_module.get_pair_latent(i, j)

            if v is not None and np.linalg.norm(v) > 1e-8:
                vecs.append(np.asarray(v, dtype=np.float64).reshape(-1))

        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                cross_nb.append(_cos(vecs[a], vecs[b]))

    ce = float(np.mean(cross_ego)) if cross_ego else 0.0
    cn = float(np.mean(cross_nb)) if cross_nb else 0.0

    return {
        "cross_ego_similarity": ce,
        "cross_neighbor_similarity": cn,
        # A value below one means same-j/different-ego representations separate
        # more strongly than different-j/same-ego representations, evidence of
        # genuine ego-centricity.
        "specificity_ratio": float(ce / (cn + 1e-8)),
        "n_cross_ego_pairs": int(len(cross_ego)),
        "n_cross_neighbor_pairs": int(len(cross_nb)),
    }
