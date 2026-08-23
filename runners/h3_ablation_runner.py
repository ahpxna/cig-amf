"""H3-specific controls built from the project's production runners."""

import torch

from models.single_mean_memory import SingleMeanPeripheral
from runners.final_runner import FinalCIGAMFRunner


class H3NoMultiMemoryRunner(FinalCIGAMFRunner):
    """Final CIG-AMF with only multi-slot aggregation replaced by one mean.

    All collection, forcing, propensities, doubly-robust proxy updates,
    signature tracking, action-time caches, belief updates, scheduler behavior,
    and policy/value training are inherited from ``FinalCIGAMFRunner``.
    """

    ablation_contract = "peripheral_multislot_to_single_mean_only"
    use_multi_memory = False

    def __init__(self, env, cfg, device="cpu"):
        super().__init__(env=env, cfg=cfg, device=device)
        self.periph_module = SingleMeanPeripheral(
            action_dim=self.action_dim,
            # These two dimensions are independently configurable only for
            # the matched-budget control.  The emitted representation keeps
            # the production ``periph_dim`` so policy inputs remain identical.
            memory_dim=cfg.get("single_mean_memory_dim", cfg["periph_memory_dim"]),
            out_dim=self.periph_dim,
            item_hidden=cfg.get("single_mean_item_hidden", 48),
            mu_floor=cfg.get("periph_mu_floor", 0.02),
            beta_floor=cfg.get("periph_beta_floor", 0.05),
            beta_mode=cfg.get("periph_beta_mode", "capacity"),
            signature_mode=cfg.get("periph_signature_mode", "full"),
            require_full_signature=cfg.get(
                "periph_require_full_signature", False
            ),
            allow_legacy_items=cfg.get("periph_allow_legacy_items", True),
        ).to(device)

        # The optimizer created by the parent references the discarded slot
        # module. Rebuild it before any training so the active single-mean
        # encoder and projection receive exactly the same policy objective.
        self.policy_optim = torch.optim.Adam(
            list(self.policy_value.parameters())
            + list(self.periph_module.parameters())
            + list(self.belief_summary_builder.parameters()),
            lr=cfg["policy_lr"],
        )
