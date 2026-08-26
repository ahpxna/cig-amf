import time
import numpy as np
import torch
import torch.nn.functional as F

from envs.causal_adapter import (
    build_dynamic_candidate_map, candidate_map_hash, resolve_env_adapter,
)
from models.structural_proxy import (
    LocalCounterfactualProxyEnsemble,
)
from models.belief_layer import BayesLightBeliefState
from models.core_behavior import PairRelationalModule
from models.peripheral_memory import PeripheralMultiMemory
from models.belief_summary import BeliefSummaryBuilder
from models.policy_value import PolicyValueNet
from training.scheduler import TwoTimescaleScheduler
from training.replay_builder import MultiEgoReplayBuilder

from models.influence_signature import InfluenceSignatureTracker
from models.intervention import EpsilonForcedActionController
from models.ego_conditioned_latent import EgoConditionedHeads
from models.drift_probe import DriftDetector, MatrixDriftDetector
from models.reciprocity import ReciprocityTracker


def _current_behavioral_phase(env):
    """Return the policy-execution phase active in the collected episode."""
    phase_fn = getattr(env, "_behaviour_mode", None)
    if callable(phase_fn):
        return str(phase_fn())
    return str(getattr(env, "mode", "unknown"))


def _safe_distribution(values, action_dim):
    """Validate and normalize a policy distribution without silent fallback."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (int(action_dim),) or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"policy distribution must have finite shape {(int(action_dim),)}, "
            f"got {arr.shape}"
        )
    if np.any(arr < 0.0):
        raise ValueError("policy distribution contains a negative probability")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("policy distribution has no positive mass")
    return (arr / total).astype(np.float32)


class FinalCIGAMFRunner:
    """
    Final CIG-AMF runner.

    Implements the current methodology:

    1. Population-wide multi-ego training:
        - every agent can be an ego agent;
        - every ego i maintains a separate belief for directed pair (i,j).

    2. Ego-centric directed influence graph:
        - belief_modules[ego] stores the structural C belief and its
          uncertainty; allocation uses the lower-confidence score G.
        - core/peripheral partitions are ego-specific.

    3. Pair-specific relational latent:
        - z_ij is not a global z_j;
        - z_ij encodes the relation between j's behaviour and ego i;
        - z_ij updates use action-selection-time context:
              o_i^t, o_j^t, a_i^t, a_j^t, Delta_ij^t.
        - collect_episode() therefore restores the pre-step snapshot before
          pair_rel_module.step_population().

    4. Shadow warm-start:
        - every pair has shadow state s_ij;
        - promotion of j into i's core warm-starts z_ij from s_ij.

    5. Local counterfactual proxy:
        - the supervised target is finite-horizon return R_i^(H);
        - proxy samples use the excluding-j context cached at policy selection:
              s_i, a_i, a_j, Z_i^{-j}, M_i^{-j}, B_i.
        - replay/proxy buffers collect every episode, including Stage 0.

    6. Bayes-light belief:
        - proxy ensemble score directed pairs.
        - belief updates use mean influence and uncertainty scale;
        - core updates use hysteresis.

    7. Two-stage two-timescale:
        - Stage 0: seeded-core warm-up, collect replay, train policy/value/pair latent.
        - Stage 1 transfers control to learned beliefs and performs periodic
          proxy/belief/core updates;
        - residual EWMA/CUSUM triggers temporarily raise structural-update frequency.

    8. Evaluation:
        - mean reward.
        - structural F1 is reported when the environment supplies diagnostic_core_by_ego.
        - temporal variance.
        - uncertainty.
        - mean core size.
        - core switches.
        - runtime / throughput.
        - proxy/bc/policy losses.
    """

    def __init__(self, env, cfg, device="cpu"):
        self.env = env
        self.env_adapter = resolve_env_adapter(env)
        self.cfg = dict(cfg)
        self.device = device

        self.n_agents = int(self.env_adapter.n_agents)
        self.obs_dim = int(self.env_adapter.obs_dim)
        self.action_dim = int(self.env_adapter.max_action_dim)
        self.candidate_max_degree = self.cfg.get("candidate_max_degree", None)
        self.candidate_refresh_interval = max(
            1, int(self.cfg.get("candidate_refresh_interval", 1))
        )
        self.candidate_cell_width = float(self.cfg.get("candidate_cell_width", 4.0))
        self.candidate_stencil_radius = int(
            self.cfg.get("candidate_stencil_radius", 1)
        )
        self.candidate_radius = self.cfg.get("candidate_radius", None)
        if self.candidate_radius is not None:
            self.candidate_radius = float(self.candidate_radius)
        self.candidate_epoch = 0
        self._last_candidate_refresh_step = -1
        self._candidate_refresh_totals = {
            "refreshes": 0, "added": 0, "removed": 0,
            "provider_work_units": 0, "refresh_ms": 0.0,
        }
        self.candidate_neighbors_by_ego, self._candidate_telemetry = (
            build_dynamic_candidate_map(
                self.env_adapter, self.candidate_max_degree,
                cell_width=self.candidate_cell_width,
                stencil_radius=self.candidate_stencil_radius,
                radius=self.candidate_radius,
            )
        )
        self.candidate_map_hash = str(
            self._candidate_telemetry["candidate_map_hash"]
        )
        self.measured_edge_count = int(self._candidate_telemetry["E_t"])
        self.candidate_construction_subquadratic = bool(
            self._candidate_telemetry.get(
                "candidate_construction_subquadratic", False
            )
        )
        self.candidate_construction_linear_candidate = bool(
            self._candidate_telemetry.get(
                "candidate_construction_linear_candidate", False
            )
        )
        self.feature_snapshot_subquadratic = bool(
            self._candidate_telemetry.get("feature_snapshot_subquadratic", False)
        )
        self.feature_snapshot_linear_candidate = bool(
            self._candidate_telemetry.get("feature_snapshot_linear_candidate", False)
        )

        self.core_dim = int(cfg["core_dim"])
        self.periph_dim = int(cfg["periph_dim"])
        self.belief_dim = int(cfg["belief_dim"])

        causal_horizon = int(cfg["causal_horizon"])
        proxy_horizon = int(cfg.get("proxy_n_horizons", causal_horizon))
        if proxy_horizon != causal_horizon:
            raise ValueError(
                "causal_horizon and proxy_n_horizons must be identical: "
                f"received {causal_horizon} and {proxy_horizon}"
            )
        confirmatory = bool(cfg.get("confirmatory", False))
        strict_causal_profile = bool(cfg.get("strict_causal_profile", False))
        structural_core_rule = str(cfg.get("belief_core_rule", "lcb")).strip().lower()
        if (confirmatory or strict_causal_profile) and (
            structural_core_rule in {"signed", "p_core"}
            or bool(cfg.get("allow_legacy_signed_core", False))
        ):
            raise ValueError(
                "Paper-contract CIG-AMF requires G=C-kappa*sigma_C "
                "allocation in the structural belief and forbids legacy "
                "signed/p_core rules; use an explicit matched-k selector "
                "ablation outside the confirmatory structural core"
            )
        cusum_threshold = cfg.get("drift_cusum_threshold")
        if cusum_threshold is None:
            if confirmatory:
                raise ValueError(
                    "confirmatory runs require a calibrated drift_cusum_threshold"
                )
            # Development-only compatibility alias.  Confirmatory manifests
            # must carry the frozen no-change calibration value explicitly.
            cusum_threshold = cfg.get("z_threshold", 8.0)
        cusum_threshold = float(cusum_threshold)
        if not np.isfinite(cusum_threshold) or cusum_threshold <= 0.0:
            raise ValueError("drift_cusum_threshold must be a positive finite value")
        if confirmatory:
            # Page-CUSUM is a calibrated tracking claim, not a detector with
            # a convenient default threshold.  The H2 launcher loads these
            # values from the frozen no-change artifact before constructing a
            # runner.  Requiring the artifact identifiers prevents a numeric
            # threshold copied from a development run from being presented as
            # confirmatory evidence.
            far_target = float(cfg.get("cusum_false_alarm_target", float("nan")))
            reference_hash = str(
                cfg.get("cusum_calibration_reference_config_hash", "")
            ).strip()
            if not (0.0 < far_target < 1.0) or not reference_hash:
                raise ValueError(
                    "confirmatory tracking requires a frozen CUSUM calibration "
                    "artifact identifier and false-alarm target"
                )
        proxy_pair_feat_dim = cfg.get("proxy_pair_feat_dim")
        if proxy_pair_feat_dim is None:
            proxy_pair_feat_dim = self.env_adapter.pair_feature_dim
        proxy_context_item_dim = cfg.get("proxy_context_item_dim")
        if proxy_context_item_dim is None:
            proxy_context_item_dim = self.env_adapter.context_item_dim
        if confirmatory or strict_causal_profile:
            if bool(cfg.get("proxy_use_belief_input", False)):
                raise ValueError(
                    "paper-locked response measurement forbids belief-summary "
                    "self-conditioning"
                )
            if int(proxy_context_item_dim) <= 0:
                raise ValueError(
                    "paper-locked response measurement requires raw partition-"
                    "independent leave-one-target context"
                )
            if int(proxy_pair_feat_dim) <= 0:
                raise ValueError(
                    "paper-locked response measurement requires target pair x_ij"
                )

        self.proxy = LocalCounterfactualProxyEnsemble(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            core_dim=self.core_dim,
            periph_dim=self.periph_dim,
            belief_dim=self.belief_dim,
            n_ensemble=cfg["n_ensemble"],
            lr=cfg["proxy_lr"],
            buffer_size=cfg.get("proxy_buffer_size", 200000),
            grad_clip=cfg.get("proxy_grad_clip", 1.0),
            device=device,
            # v2 defaults match structural_proxy.py and preserve behaviour
            # when the configuration omits them.
            n_horizons=proxy_horizon,
            discount=cfg["discount"],
            effect_mode=cfg.get(
                "proxy_effect_mode", "signed_policy_contrast"
            ),
            # Row-level AIPW is a diagnostic estimator only.  The online
            # signature path is the plug-in C/D estimator specified by the
            # papers; never enable DR merely because a custom config omitted
            # the key.
            use_doubly_robust=cfg.get("proxy_use_doubly_robust", False),
            iw_clip=cfg.get("proxy_iw_clip", 10.0),
            bootstrap_ratio=cfg.get("proxy_bootstrap_ratio", 0.8),
            use_belief_input=cfg.get("proxy_use_belief_input", False),
            ensemble_dropout=cfg.get("proxy_ensemble_dropout", 0.0),
            seed=cfg.get("seed", 0),
            # [FIX-X1] x_ij from Eq. 8. Set zero for the legacy no-x_ij
            # ablation, the configuration that failed H1 over eight seeds.
            pair_feat_dim=proxy_pair_feat_dim,
            context_item_dim=proxy_context_item_dim,
            debug_verbose=cfg.get("debug_verbose", False),
            forced_only_training=cfg.get("proxy_forced_only_training", False),
            response_ipw_ablation=cfg.get("proxy_response_ipw_ablation", False),
        )

        self.pair_rel_module = PairRelationalModule(
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.core_dim,
            shadow_dim=cfg["shadow_dim"],
            rel_feat_dim=int(getattr(self.env_adapter, "relation_feature_dim", 6)),
            lr=cfg["core_lr"],
            bc_buffer_size=cfg.get("bc_buffer_size", 200000),
            grad_clip=cfg.get("bc_grad_clip", 1.0),
            shadow_loss_weight=cfg.get("shadow_loss_weight", 0.25),
            state_mode=cfg.get("pair_state_mode", "recurrent"),
            candidate_neighbors_by_ego=self.candidate_neighbors_by_ego,
            device=device,
        )

        self.periph_module = PeripheralMultiMemory(
            action_dim=self.action_dim,
            num_slots=cfg["num_memory_slots"],
            memory_dim=cfg["periph_memory_dim"],
            out_dim=self.periph_dim,
            mu_floor=cfg.get("periph_mu_floor", 0.0),
            beta_floor=cfg.get("periph_beta_floor", 0.0),
            beta_mode=cfg.get("periph_beta_mode", "capacity"),
            semantic_mass=cfg.get("periph_semantic_mass", 0.5),
            lambda_sigma=cfg.get("periph_lambda_sigma", 1.0),
            use_uniform_mix=cfg.get("periph_use_uniform_mix", False),
            uniform_mix=cfg.get("periph_uniform_mix", 0.0),
            lb_coeff=cfg.get("periph_lb_coeff", 1e-2),
            tau_role=cfg.get("periph_tau_D", 0.05),
            sigma_hi=cfg.get("periph_sigma_D_hi", 0.5),
            temperature_D=cfg.get("periph_temperature_D", 0.05),
            temperature_0=cfg.get("periph_temperature_0", 0.05),
            temperature_sigma=cfg.get("periph_temperature_sigma", 0.05),
            # [FIX-2] orth_coeff previously ignored cfg and remained at 1e-2.
            # The No-AuxLoss ablation then disabled only L_lb while L_orth
            # remained active, failing to isolate Eq. 26-27 L_aux.
            orth_coeff=cfg.get("periph_orth_coeff", 1e-2),
            routing_mode=cfg.get("periph_routing_mode", "semantic"),
            signature_mode=cfg.get("periph_signature_mode", "full"),
            require_full_signature=cfg.get(
                "periph_require_full_signature", True
            ),
            allow_legacy_items=cfg.get("periph_allow_legacy_items", False),
        ).to(device)
        if (confirmatory or strict_causal_profile) and (
            bool(self.periph_module.allow_legacy_items)
            or bool(self.periph_module.use_uniform_mix)
        ):
            raise ValueError(
                "Paper-contract CIG-AMF requires retained typed C/D profiles "
                "and disables uniform-memory compatibility mixing"
            )

        self.belief_summary_builder = BeliefSummaryBuilder(
            top_k=cfg["belief_top_k"],
            pooled_hidden=cfg["belief_pooled_hidden"],
            out_dim=self.belief_dim,
            priority_mu_floor=cfg.get("belief_priority_mu_floor", 0.0),
        ).to(device)

        self.policy_value = PolicyValueNet(
            obs_dim=self.obs_dim,
            core_dim=self.core_dim,
            peripheral_dim=self.periph_dim,
            belief_dim=self.belief_dim,
            action_dim=self.action_dim,
            hidden=cfg["policy_hidden"],
        ).to(device)
        if bool(cfg.get("freeze_downstream_policy_value", False)):
            for parameter in self.policy_value.parameters():
                parameter.requires_grad_(False)
        if bool(cfg.get("freeze_belief_summary_learning", False)):
            # Representation-isolation panels hold the common structural
            # summary fixed.  Otherwise a periphery comparison can change
            # both M_i and the learned B_i projection, invalidating the
            # claimed attribution of decision-fidelity differences.
            for parameter in self.belief_summary_builder.parameters():
                parameter.requires_grad_(False)

        self.policy_optim = torch.optim.Adam(
            list(self.policy_value.parameters())
            + list(self.periph_module.parameters())
            + list(self.belief_summary_builder.parameters()),
            lr=cfg["policy_lr"],
        )

        self.scheduler = TwoTimescaleScheduler(
            k0_warmup=cfg["k0_warmup"],
            alpha_fast=cfg.get("policy_lr", 1e-3),
            alpha_slow_ratio=cfg.get("slow_ratio", 0.05),
            accel_factor=cfg.get("accel_factor", 4.0),
            accel_duration=cfg.get("accel_duration", 8),
            z_threshold=cusum_threshold,
            require_both=cfg.get("require_both", False),
            refractory=cfg.get("refractory", 10),
            inflation_factor=cfg.get("inflation_factor", 2.5),
            inflation_t_reset=cfg.get("inflation_t_reset", 1),
        )
        self.sig_tracker = InfluenceSignatureTracker(
            n_agents=self.n_agents,
            window=cfg.get("sig_tracker_window", 30),
            direction_window=cfg.get("sig_tracker_direction_window", 5),
        )
        # C/D tracker values are slow targets for Eq. (14), not instantaneous
        # pair labels. Their action-time age is measured in environment steps
        # and stale supervision is masked by a configured step bound.
        self._profile_update_step = {}
        self.forcer = EpsilonForcedActionController(
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            eps=cfg.get("eps", 0.03),
            # [FIX-P1] Default None: a cap invalidates the exact-known-b_j
            # property; see intervention.py. It was almost never reached at
            # eps=0.03 with n=24.
            max_forced_per_step=cfg.get("forcer_max_forced_per_step", None),
            anneal_to=cfg.get("forcer_anneal_to", 0.01),
            anneal_episodes=cfg.get("forcer_anneal_episodes", 60),
            rng=np.random.RandomState(cfg.get("seed", 0)),
        )
        if (confirmatory or strict_causal_profile) and cfg.get(
            "forcer_max_forced_per_step"
        ) is not None:
            raise ValueError(
                "Paper-contract causal runs cannot cap simultaneous forcing: "
                "the cap changes the known behaviour propensity"
            )
        self.heads = EgoConditionedHeads(
            latent_dim=self.pair_rel_module.hidden_dim,
        ).to(self.device)
        self.heads_optim = torch.optim.Adam(
            self.heads.parameters(), lr=cfg.get("heads_lr", cfg.get("core_lr", 5e-4))
        )
        self.drift = DriftDetector(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_horizons=causal_horizon,
            discount=cfg["discount"],
            warmup_batches=cfg.get("drift_warmup_batches", 200),
            batch_size=cfg.get("drift_batch_size", 256),
            recalibrate_after=cfg.get("drift_recalibrate_after", 15),
            window=cfg.get("drift_window", 20),
            seed=cfg.get("seed", 0),
            device=self.device,
            cusum_allowance=cfg.get("drift_cusum_allowance", 0.5),
            cusum_threshold=cusum_threshold,
        )
        self.matdet = MatrixDriftDetector(window=cfg.get("matdet_window", 20))
        self.recip = ReciprocityTracker(
            n_agents=self.n_agents,
            min_causal_samples=cfg.get("recip_min_causal_samples", 20),
        )

        self.replay_builder = MultiEgoReplayBuilder(
            discount=cfg["discount"],
            horizon=causal_horizon,
        )
        if not (
            self.proxy.n_horizons
            == self.drift.n_horizons
            == self.replay_builder.horizon
            == causal_horizon
        ):
            raise RuntimeError(
                "causal horizon contract violated across proxy, drift, and replay"
            )

        full_explicit_mode = (
            str(self.cfg.get("core_selection_mode", "structural_capacity"))
            .strip().lower() == "full_explicit"
        )
        self.belief_modules = {
            ego: BayesLightBeliefState(
                ego_id=ego,
                neighbor_ids=self.candidate_neighbors_by_ego[ego],
                lambda_0=cfg["belief_lambda_0"],
                uncertainty_scale=cfg["belief_uncertainty_scale"],
                tau=cfg.get(
                    "belief_tau_enter",
                    cfg.get("belief_tau_in", cfg.get("belief_tau", 0.005)),
                ),
                tau_in=cfg.get(
                    "belief_tau_enter",
                    cfg.get("belief_tau_in", cfg.get("belief_tau", 0.005)),
                ),
                tau_out=cfg.get(
                    "belief_tau_out",
                    cfg.get("belief_tau_exit", 0.00175),
                ),
                weak_prior_top_k=cfg["seed_core_top_k"],
                min_core_size=(
                    self.n_agents - 1 if full_explicit_mode
                else cfg.get("min_core_size", 2)
                ),
                max_core_size=(
                    self.n_agents - 1 if full_explicit_mode
                else cfg.get("max_core_size", 4)
                ),
                sigma_floor=cfg.get("sigma_floor", 0.08),
                # v2 defaults match belief_layer.py recommendations and
                # preserve behaviour when cfg omits them.
                core_rule=cfg.get("belief_core_rule", "lcb"),
                kappa=cfg.get("belief_kappa", 1.0),
                alpha_decay=cfg.get("belief_alpha_decay", 0.7),
                adaptive_k=(
                    False if full_explicit_mode
                    else cfg.get("belief_adaptive_k", True)
                ),
                adaptive_k_min=(
                    self.n_agents - 1 if full_explicit_mode
                else cfg.get(
                        "belief_adaptive_k_min", cfg.get("min_core_size", 2)
                    )
                ),
                signed_balance=cfg.get("belief_signed_balance", 0.5),
                sigma_alpha_max=cfg.get("belief_sigma_alpha_max", 1.0),
                allow_legacy_signed_core=cfg.get(
                    "allow_legacy_signed_core", False
                ),
            )
            for ego in range(self.n_agents)
        }

        self._initialize_seeded_cores()
        if full_explicit_mode:
            for belief in self.belief_modules.values():
                belief.max_core_size = len(belief.neighbor_ids)
                belief.min_core_size = len(belief.neighbor_ids)
                belief.adaptive_k = False
                belief.adaptive_k_min = len(belief.neighbor_ids)
                belief.set_fixed_core(belief.neighbor_ids)
        # Seeded belief cores are already part of the policy representation.
        # Reconcile them immediately so the explicit pair tier is live from
        # episode zero rather than waiting for the first slow graph update.
        self.pair_rel_module.reconcile_core_sets(
            {ego: belief.get_core_set() for ego, belief in self.belief_modules.items()},
            warm_start=bool(self.cfg.get("pair_warm_start", True)),
        )
        self._selector_random_rngs = {}

        self.history = {
            "episodes": [],
            "mean_reward": [],
            "reward_per_agent": [],
            "mean_f1": [],
            "mean_f1_role": [],   # [F1-TOPK] control score against role labels
            "mean_temporal_var": [],
            "mean_uncertainty": [],
            "mean_core_size": [],
            "mean_core_switches": [],
            "mean_capacity": [],
            "max_g": [],
            "mean_mu": [],
            "max_p": [],
            "runtime": [],
            "throughput_agent_steps_per_sec": [],
            "episode_runtime_total": [],
            "throughput_total_agent_steps_per_sec": [],
            "proxy_train_residual": [],
            "proxy_holdout_residual": [],
            "scheduler_residual_ewma": [],
            "scheduler_cusum_score": [],
            "drift_monitoring_ready": [],
            "drift_phase": [],
            "scheduler_accel_remaining": [],
            "proxy_loss": [],
            "bc_loss": [],
            "policy_loss": [],
            # [debug doc section 1.2] The old policy_loss combined actor,
            # critic, and entropy and could not identify the failing component.
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "grad_norm_preclip": [],
            "adv_mean": [],
            "adv_std": [],
            "triggered": [],
            "trigger_count": [],
            "stage": [],
            "proxy_buffer_size": [],
            "pushed_proxy_samples": [],
            "promoted": [],
            "demoted": [],
            "measured_edge_count": [],
            "candidate_max_degree": [],
            "candidate_construction_subquadratic": [],
            "candidate_construction_linear_candidate": [],
            "feature_snapshot_subquadratic": [],
            "feature_snapshot_linear_candidate": [],
            "candidate_epoch": [],
            "candidate_map_hash": [],
            "candidate_provider": [],
            "candidate_refresh_ms": [],
            "candidate_provider_work_units": [],
            "candidate_cell_occupancy_mean": [],
            "candidate_cell_occupancy_max": [],
            "candidate_added_pairs": [],
            "candidate_removed_pairs": [],
            "candidate_churn": [],
            "E_t": [],
            "K_t": [],
            "d_bar": [],
            "k_bar": [],
            "active_core_pair_count": [],
            "shadow_pair_count": [],
            "full_state_pair_count": [],
            "belief_pair_count": [],
            "signature_pair_count": [],
            "degree_limited_ego_count": [],
            "causal_latency_valid_fraction": [],
            "causal_latency_onset_valid_fraction": [],
            "causal_latency_cm_mean": [],
        }

        # ``run`` is intentionally re-entrant because the H2/H3 protocols
        # inspect the runner at fixed evaluation boundaries.  Episode numbers
        # therefore belong to the runner, not to an individual ``run`` call.
        # ``episode_events`` retains the per-episode event stream even when
        # scalar metrics are sampled less frequently through ``eval_every``.
        self.episodes_completed = 0
        self._interaction_step = 0
        # Version every causal replay row.  ``structure_regime_id`` is owned
        # by the environment when it exposes one; the runner never fabricates
        # a structural change from a detector alarm.  ``policy_version`` is
        # incremented after each learner update.
        self._policy_version = 0
        self.episode_events = []
        self._last_behavioral_phase = None
        self._latest_direction_matrix = np.zeros(
            (self.n_agents, self.n_agents), dtype=np.float64
        )

    # ============================================================
    # Construction helpers
    # ============================================================

    def _current_structure_regime_id(self) -> int:
        """Return the environment-declared structural regime, or base regime.

        A Page-CUSUM trigger is evidence for adaptation, not ground truth that
        the environment mechanism changed.  It therefore must not mutate this
        identifier.  Adapters/environments that expose a regime counter can
        provide ``get_structure_regime_id()`` or ``structure_regime_id``.
        """
        for owner in (self.env_adapter, self.env):
            getter = getattr(owner, "get_structure_regime_id", None)
            value = getter() if callable(getter) else getattr(
                owner, "structure_regime_id", None
            )
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "structure_regime_id must be an integer when declared"
                ) from exc
            if not np.isfinite(numeric) or int(numeric) != numeric or numeric < 0:
                raise ValueError(
                    "structure_regime_id must be a non-negative finite integer"
                )
            return int(numeric)
        return 0

    def _make_belief_state(self, ego):
        """
        Create BayesLightBeliefState.

        Construct the same C-only belief contract used during normal runner
        initialization.  This helper remains public for ablation code; it
        must not silently recreate the retired ratio-based/signed legacy
        configuration.
        """
        full_explicit_mode = (
            str(self.cfg.get("core_selection_mode", "structural_capacity"))
            .strip().lower() == "full_explicit"
        )
        return BayesLightBeliefState(
            ego_id=ego,
            neighbor_ids=self._candidate_ids(ego),
            lambda_0=self.cfg["belief_lambda_0"],
            uncertainty_scale=self.cfg["belief_uncertainty_scale"],
            tau=self.cfg.get("belief_tau_enter", self.cfg.get("belief_tau_in", 0.005)),
            tau_in=self.cfg.get("belief_tau_enter", self.cfg.get("belief_tau_in", 0.005)),
            tau_out=self.cfg.get("belief_tau_exit", self.cfg.get("belief_tau_out", 0.00175)),
            weak_prior_top_k=self.cfg["seed_core_top_k"],
            min_core_size=(
                self.n_agents - 1 if full_explicit_mode
                else self.cfg.get("min_core_size", 2)
            ),
            max_core_size=(
                self.n_agents - 1 if full_explicit_mode
                else self.cfg.get("max_core_size", 4)
            ),
            sigma_floor=self.cfg.get("sigma_floor", 0.08),
            core_rule=self.cfg.get("belief_core_rule", "lcb"),
            kappa=self.cfg.get("belief_kappa", 1.0),
            alpha_decay=self.cfg.get("belief_alpha_decay", 0.7),
            adaptive_k=(
                False if full_explicit_mode
                else self.cfg.get("belief_adaptive_k", True)
            ),
            adaptive_k_min=(
                self.n_agents - 1 if full_explicit_mode
                else self.cfg.get("belief_adaptive_k_min", self.cfg.get("min_core_size", 2))
            ),
            signed_balance=self.cfg.get("belief_signed_balance", 0.5),
            sigma_alpha_max=self.cfg.get("belief_sigma_alpha_max", 1.0),
            allow_legacy_signed_core=self.cfg.get("allow_legacy_signed_core", False),
        )

    # ============================================================
    # Stage 0 weak structural prior
    # ============================================================

    def _compute_weak_prior_scores(self, ego):
        """
        Weak structural prior, not ground truth.

        Stage 0 may use:
        - proximity.
        - same-zone bonus.
        - role relevance.
        - environment-specific local cues.

        This prior only seeds the core during warm-up. In Stage 1, learned
        beliefs determine the core/peripheral partition.
        """
        scores = {}

        for j in self._candidate_ids(ego):

            scores[j] = float(self.env_adapter.weak_prior_score(ego, j))

        return scores

    def _initialize_seeded_cores(self):
        for ego in range(self.n_agents):
            prior_scores = self._compute_weak_prior_scores(ego)
            self.belief_modules[ego].initialize_from_weak_prior(prior_scores)

    def _reset_switch_counters_if_available(self):
        """
        Use after the Stage 0 to Stage 1 transition.

        Stage 0 may reseed the core each episode from the weak prior. Position
        resets can change that prior, contaminating core-switch counts with
        bootstrap variation rather than learned-belief instability. Call
        reset_switch_counter() when available and otherwise retain compatibility.
        """
        for ego in range(self.n_agents):
            mod = self.belief_modules[ego]
            if hasattr(mod, "reset_switch_counter"):
                mod.reset_switch_counter()

    # ============================================================
    # Belief / core / peripheral summaries
    # ============================================================

    def _candidate_ids(self, ego):
        """Measured neighbour set for the current candidate epoch."""
        return list(self.candidate_neighbors_by_ego[int(ego)])

    def _refresh_dynamic_candidates(self, *, force=False):
        """Refresh the pre-measurement candidate interface and live pair state.

        Retained pairs preserve online state. Added pairs start with neutral/high-
        uncertainty belief and zero shadow state. Removed pairs are demoted and
        evicted from live pair-indexed modules so persistent memory follows E_t.
        Historical response replay remains timestamped evidence and is untouched.
        """
        step = int(getattr(self, "_interaction_step", 0))
        if (
            not bool(force)
            and self._last_candidate_refresh_step >= 0
            and step - self._last_candidate_refresh_step < self.candidate_refresh_interval
        ):
            return dict(self._candidate_telemetry)
        new_map, telemetry = build_dynamic_candidate_map(
            self.env_adapter, self.candidate_max_degree,
            cell_width=self.candidate_cell_width,
            stencil_radius=self.candidate_stencil_radius,
            radius=self.candidate_radius,
        )
        old_map = self.candidate_neighbors_by_ego
        old_pairs = {(int(e), int(j)) for e, ids in old_map.items() for j in ids}
        new_pairs = {(int(e), int(j)) for e, ids in new_map.items() for j in ids}
        added = new_pairs - old_pairs
        removed = old_pairs - new_pairs
        union = old_pairs | new_pairs
        churn = float(len(added) + len(removed)) / float(max(1, len(union)))

        self.pair_rel_module.reconcile_candidate_sets(new_map)
        for ego in range(self.n_agents):
            self.belief_modules[ego].reconcile_neighbors(new_map[int(ego)])
        self.sig_tracker.reconcile_candidate_pairs(new_map)
        for key in list(self._profile_update_step.keys()):
            if key not in new_pairs:
                self._profile_update_step.pop(key, None)
        self.candidate_neighbors_by_ego = {
            int(ego): tuple(int(j) for j in ids) for ego, ids in new_map.items()
        }
        self.pair_rel_module.reconcile_core_sets(
            {ego: belief.get_core_set() for ego, belief in self.belief_modules.items()},
            warm_start=bool(self.cfg.get("pair_warm_start", True)),
        )
        self._reset_exclusion_caches()

        self.candidate_epoch += 1
        self._last_candidate_refresh_step = step
        self.candidate_map_hash = str(telemetry["candidate_map_hash"])
        self.measured_edge_count = int(telemetry["E_t"])
        self.candidate_construction_subquadratic = bool(
            telemetry.get("candidate_construction_subquadratic", False)
        )
        self.candidate_construction_linear_candidate = bool(
            telemetry.get("candidate_construction_linear_candidate", False)
        )
        self.feature_snapshot_subquadratic = bool(
            telemetry.get("feature_snapshot_subquadratic", False)
        )
        self.feature_snapshot_linear_candidate = bool(
            telemetry.get("feature_snapshot_linear_candidate", False)
        )
        telemetry = dict(telemetry)
        telemetry.update({
            "candidate_epoch": int(self.candidate_epoch),
            "candidate_added_pairs": int(len(added)),
            "candidate_removed_pairs": int(len(removed)),
            "candidate_churn": float(churn),
        })
        self._candidate_telemetry = telemetry
        totals = self._candidate_refresh_totals
        totals["refreshes"] += 1
        totals["added"] += len(added)
        totals["removed"] += len(removed)
        totals["provider_work_units"] += int(telemetry.get("provider_work_units", 0))
        totals["refresh_ms"] += float(telemetry.get("candidate_refresh_ms", 0.0))
        return dict(telemetry)

    def _build_belief_items_for_ego(self, ego):
        belief_state = self.belief_modules[ego].get_state_dict()
        items = []

        for j in self._candidate_ids(ego):

            pair_norm = np.linalg.norm(
                self.pair_rel_module.get_pair_latent(ego, j)
            )

            items.append(
                self.belief_summary_builder.build_item(
                    ego_id=ego,
                    j=j,
                    env=self.env,
                    belief_state=belief_state,
                    pair_latent_norm=pair_norm,
                )
            )

        if len(items) == 0:
            return np.zeros(
                (0, int(self.belief_summary_builder.item_dim)), dtype=np.float32
            )

        return np.stack(items, axis=0).astype(np.float32)

    def _belief_summary_tensor_from_items(self, items_np):
        return self.belief_summary_builder(items_np)

    def _belief_summary_np_from_items(self, items_np):
        with torch.no_grad():
            return (
                self._belief_summary_tensor_from_items(items_np)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    def _core_summary_for_ego(self, ego):
        core_set = self.belief_modules[ego].get_core_set()
        return self.pair_rel_module.get_core_summary(ego, core_set)

    def _build_periph_inputs_for_ego(self, ego, causal_pair_signals=None):
        belief_mod = self.belief_modules[ego]
        periph_ids = sorted(list(belief_mod.get_peripheral_set()))
        belief_state = belief_mod.get_state_dict()

        return self.periph_module.build_inputs(
            ego_id=ego,
            peripheral_ids=periph_ids,
            env=self.env,
            belief_state=belief_state,
            prev_core_set=belief_mod.prev_core_set,
            influence_signatures={
                int(j): self.sig_tracker.get_signature(ego, j)
                for j in periph_ids
            },
            context_validity={
                int(j): self.sig_tracker.get_context_validity(ego, j)
                for j in periph_ids
            },
            causal_pair_signals=(
                None
                if causal_pair_signals is None
                else {
                    int(j): causal_pair_signals[int(j)]
                    for j in periph_ids
                    if int(j) in causal_pair_signals
                }
            ),
        )

    def _cd_target_for_pair(self, ego_id, neighbor_id):
        """Return timestamped slow C/D supervision for the pair encoder."""
        ego_id = int(ego_id)
        neighbor_id = int(neighbor_id)
        target_step = self._profile_update_step.get((ego_id, neighbor_id))
        current_step = int(getattr(self, "_interaction_step", 0))
        if target_step is None:
            age = 10**9
            target_timestamp = -1
        else:
            target_step = int(target_step)
            age = max(0, current_step - target_step)
            target_timestamp = target_step
        valid_action_count = int(
            np.count_nonzero(self.env_adapter.valid_action_mask(neighbor_id))
        )
        signal = self.sig_tracker.get_pair_signal(
            ego_id,
            neighbor_id,
            timestamp=target_timestamp,
            structure_regime_id=self._current_structure_regime_id(),
            support_valid=bool(target_step is not None and valid_action_count >= 2),
            valid_action_count=valid_action_count,
        )
        return {
            "target": signal.allocator_profile[:2],
            "age": int(age),
            "timestamp": int(target_timestamp),
            # A C/D label is usable only when it came from an actual scored
            # response surface.  The age threshold is applied by
            # PairRelationalModule; this flag prevents its initial zero state
            # from becoming fabricated supervision.
            "valid": bool(signal.support_valid and target_step is not None),
        }

    def _periph_summary_tensor_from_inputs(self, inputs):
        return self.periph_module(inputs)
    
    def _periph_full_from_inputs(self, inputs):
        """Full path used in training so aux_loss retains gradients."""
        return self.periph_module.forward_full(inputs)
    
    def _periph_summary_np_from_inputs(self, inputs):
        with torch.no_grad():
            return (
                self._periph_summary_tensor_from_inputs(inputs)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    def _core_context_excluding(self, ego, exclude_j):
        """
        [GPU_OPTIMIZATION_CONTRACT.md section 2.1] The old path recomputed
        get_core_summary() from scratch for every (ego,j), repeating
        O(core_size) work N times per ego and N^2 times per timestep.
        core_behavior.py already provides get_core_summary_excluding_all via a
        sum-minus-one identity valid specifically for mean pooling, the paper's
        pooling method. Cache by (ego,current core_set): one operation computes
        all N neighbours and subsequent dictionary lookups are O(1).
        """
        core_set = self.belief_modules[ego].get_core_set()
        cache_key = (ego, frozenset(core_set))

        if getattr(self, "_core_excl_cache_key", None) != cache_key:
            self._core_excl_cache_key = cache_key
            self._core_excl_cache = (
                self.pair_rel_module.get_core_summary_excluding_all(ego, core_set)
            )

        if exclude_j in self._core_excl_cache:
            return self._core_excl_cache[exclude_j]

        # Safe legacy slow-path fallback; exclude_j should always be valid, but
        # correctness takes priority if that invariant is violated.
        reduced = [x for x in core_set if x != exclude_j]
        return self.pair_rel_module.get_core_summary(ego, reduced)

    def _reset_exclusion_caches(self):
        """Invalidate action-time contexts whenever state or signatures move."""
        self._core_excl_cache_key = None
        self._core_excl_cache = {}
        self._periph_excl_cache_key = None
        self._periph_excl_cache = {}

    def _periph_context_excluding(self, ego, exclude_j):
        """
        [GPU_OPTIMIZATION_CONTRACT.md section 2.1] The old path ran full
        build_inputs and forward, including item_encoder/slot_router, once per
        (ego,j). forward_excluding_all computes the full numerator/denominator
        once and vectorizes sum-minus-one across N neighbours. The identity is
        valid for the paper's weighted-mean pooling. Cache by (ego,current
        peripheral_set) and synchronize CPU once for the batch rather than once
        per excluded j.
        """
        belief_mod = self.belief_modules[ego]
        periph_ids = sorted(list(belief_mod.get_peripheral_set()))
        cache_key = (ego, tuple(periph_ids))

        if getattr(self, "_periph_excl_cache_key", None) != cache_key:
            self._periph_excl_cache_key = cache_key
            if len(periph_ids) == 0:
                self._periph_excl_cache = {}
            else:
                belief_state = belief_mod.get_state_dict()
                inputs = self.periph_module.build_inputs(
                    ego_id=ego,
                    peripheral_ids=periph_ids,
                    env=self.env,
                    belief_state=belief_state,
                    prev_core_set=belief_mod.prev_core_set,
                    influence_signatures={
                        int(j): self.sig_tracker.get_signature(ego, j)
                        for j in periph_ids
                    },
                    context_validity={
                        int(j): self.sig_tracker.get_context_validity(ego, j)
                        for j in periph_ids
                    },
                )
                raw = self.periph_module.forward_excluding_all(inputs, periph_ids)
                if raw:
                    ids_order = list(raw.keys())
                    stacked = (
                        torch.stack([raw[j] for j in ids_order], dim=0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    self._periph_excl_cache = {
                        j: stacked[i] for i, j in enumerate(ids_order)
                    }
                else:
                    self._periph_excl_cache = {}

        if exclude_j in self._periph_excl_cache:
            return self._periph_excl_cache[exclude_j]

        # Safe legacy slow-path fallback; exclude_j should be peripheral, but
        # never return an incorrect result if the invariant is violated.
        periph_ids_reduced = sorted(
            list(belief_mod.get_peripheral_set() - {exclude_j})
        )
        belief_state = belief_mod.get_state_dict()
        inputs = self.periph_module.build_inputs(
            ego_id=ego,
            peripheral_ids=periph_ids_reduced,
            env=self.env,
            belief_state=belief_state,
            prev_core_set=belief_mod.prev_core_set,
            influence_signatures={
                int(j): self.sig_tracker.get_signature(ego, j)
                for j in periph_ids_reduced
            },
            context_validity={
                int(j): self.sig_tracker.get_context_validity(ego, j)
                for j in periph_ids_reduced
            },
        )
        return self._periph_summary_np_from_inputs(inputs)

    @staticmethod
    def _fit_raw_context(values, width):
        """Place a fixed raw-set summary in a declared proxy context width."""
        out = np.zeros(int(width), dtype=np.float32)
        source = np.asarray(values, dtype=np.float32).reshape(-1)
        out[:min(out.size, source.size)] = source[:out.size]
        return out

    def _split_raw_context(self, values):
        """Pack a raw set summary across both legacy context tensors."""
        packed = self._fit_raw_context(
            values,
            int(self.core_dim) + int(self.periph_dim),
        )
        return packed[:self.core_dim], packed[self.core_dim:]

    def _raw_proxy_context_excluding(self, ego, exclude_j, current_actions=None):
        """Build a partition-independent leave-one-neighbour context.

        This measurement input is derived only from current observable
        geometry, zones, and current executed actions. It deliberately does not use
        the learned core, peripheral memory, pair latent, or belief state, so
        the path is ``proxy -> partition -> policy`` rather than a feedback
        loop from an earlier partition into the proxy target.
        """
        adapter = getattr(self, "env_adapter", None)
        if adapter is None:
            adapter = resolve_env_adapter(self.env, action_dim=self.action_dim)
        table, mask = self._raw_proxy_context_items_excluding(
            ego, exclude_j, current_actions=current_actions
        )
        active = table[np.asarray(mask, dtype=bool)]
        if active.size == 0:
            summary = np.zeros(
                2 * int(adapter.context_item_dim) + 1,
                dtype=np.float32,
            )
        else:
            summary = np.concatenate([
                active.sum(axis=0), np.square(active).sum(axis=0), [len(active)]
            ]).astype(np.float32)
        return self._split_raw_context(summary)

    def _raw_proxy_context_items_excluding(
        self, ego, exclude_j, current_actions=None
    ):
        """Return raw neighbour items for literal leave-one-out DeepSets."""
        adapter = getattr(self, "env_adapter", None)
        if adapter is None:
            adapter = resolve_env_adapter(self.env, action_dim=self.action_dim)
        ego = int(ego)
        exclude_j = int(exclude_j)
        action_source = (
            current_actions
            if current_actions is not None
            else getattr(self.env, "last_actions", {})
        )
        rows = []
        for other in self._candidate_ids(ego):
            if other == exclude_j:
                continue
            try:
                action_index = int(action_source[other])
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(
                    f"raw causal context is missing the co-action for neighbour {other}"
                ) from exc
            if action_index < 0 or action_index >= self.action_dim:
                raise ValueError(
                    f"raw causal context contains invalid co-action {action_index} "
                    f"for neighbour {other}"
                )
            rows.append(
                adapter.neighbour_features(ego, other, action_index)
            )
        if not rows:
            return (
                np.zeros((1, adapter.context_item_dim), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
            )
        table = np.stack(rows, axis=0).astype(np.float32)
        if table.shape[1] != adapter.context_item_dim:
            raise RuntimeError(
                "environment adapter context item dimension mismatch: "
                f"declared {adapter.context_item_dim}, got {table.shape[1]}"
            )
        return table, np.ones((table.shape[0],), dtype=np.float32)

    def _raw_proxy_context_block(self, ego, current_actions=None):
        """Build the raw DeepSets block once for one ego at one timestep.

        Targets retain only a reference to this block.  The proxy forms the
        literal leave-one-out set with ``S_i - e_ij`` rather than materialising
        an O(N)-sized context table for every target j.
        """
        adapter = self.env_adapter
        action_source = (
            current_actions if current_actions is not None
            else getattr(self.env, "last_actions", {})
        )
        ids = self._candidate_ids(ego)
        rows = []
        for other in ids:
            try:
                action = int(action_source[other])
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(
                    f"raw causal context block is missing the co-action for neighbour {other}"
                ) from exc
            if action < 0 or action >= self.action_dim:
                raise ValueError(
                    f"raw causal context block contains invalid co-action {action} "
                    f"for neighbour {other}"
                )
            rows.append(adapter.neighbour_features(ego, other, action))
        items = (
            np.stack(rows, axis=0).astype(np.float32)
            if rows else np.zeros((0, adapter.context_item_dim), dtype=np.float32)
        )
        return {
            "context_block_id": (
                int(getattr(self, "_interaction_step", 0)) * max(1, self.n_agents)
                + int(ego)
            ),
            "neighbor_ids": np.asarray(ids, dtype=np.int64),
            "items": items,
        }

    def _raw_proxy_context_excluding_from_block(self, block, exclude_j):
        """Legacy fixed-width context summary derived by sum-minus-one."""
        items = np.asarray(block["items"], dtype=np.float32)
        ids = np.asarray(block["neighbor_ids"], dtype=np.int64)
        keep = ids != int(exclude_j)
        active = items[keep]
        width = int(self.env_adapter.context_item_dim)
        if active.size == 0:
            summary = np.zeros(2 * width + 1, dtype=np.float32)
        else:
            summary = np.concatenate([
                active.sum(axis=0), np.square(active).sum(axis=0), [len(active)]
            ]).astype(np.float32)
        return self._split_raw_context(summary)

    # ============================================================
    # Action selection
    # ============================================================

    def _behavioural_execution_distribution(self, agent_id):
        """Return the environment's scripted execution distribution.

        The H2 behavioural arm must alter executed actions, not merely a
        diagnostic label in the environment.  ``OmniArena`` provides this
        distribution without sampling so that querying the intervention does
        not perturb the environment RNG.  Older environments may expose only
        a deterministic ``scripted_policy``; that compatibility path is safe
        only for deterministic scripted policies.
        """
        distribution_fn = getattr(self.env, "scripted_policy_distribution", None)
        if callable(distribution_fn):
            return _safe_distribution(distribution_fn(int(agent_id)), self.action_dim)

        policy_fn = getattr(self.env, "scripted_policy", None)
        if callable(policy_fn):
            action = int(policy_fn(int(agent_id)))
            if action < 0 or action >= self.action_dim:
                raise ValueError("scripted policy returned an invalid action")
            out = np.zeros(self.action_dim, dtype=np.float32)
            out[action] = 1.0
            return out

        return np.full(self.action_dim, 1.0 / float(self.action_dim), dtype=np.float32)

    def _execution_policy_adapter(self, learned_probs, valid_action_masks=None):
        """Mix learned and scripted policy distributions for controlled H2.

        The returned distribution is the policy that is actually sampled
        before epsilon forcing.  It is therefore also the target ``pi`` used
        for the directional contrast D and the base distribution for the
        logged behaviour propensity b.  The adapter is disabled by default
        and enabled only by the controlled behavioural-drift protocol.
        """
        learned = np.asarray(learned_probs, dtype=np.float32)
        expected = (self.n_agents, self.action_dim)
        if learned.shape != expected:
            raise ValueError(
                f"learned policy matrix must have shape {expected}, got {learned.shape}"
            )
        if valid_action_masks is None:
            valid = np.ones_like(learned, dtype=bool)
        else:
            valid = np.asarray(valid_action_masks, dtype=bool)
            if valid.shape != expected or np.any(valid.sum(axis=1) == 0):
                raise ValueError(
                    "valid_action_masks must match the policy matrix and "
                    "retain at least one action per agent"
                )

        def _masked_distribution(row, mask):
            row = np.asarray(row, dtype=np.float32).reshape(-1)
            if row.shape != (self.action_dim,) or not np.all(np.isfinite(row)):
                raise ValueError("policy distribution has invalid shape or values")
            if np.any(row < 0.0):
                raise ValueError("policy distribution contains a negative probability")
            masked = np.where(mask, row, 0.0)
            total = float(masked.sum())
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "policy distribution assigns no mass to the state's valid actions"
                )
            return masked / float(masked.sum())

        learned = np.stack([
            _masked_distribution(row, valid[agent])
            for agent, row in enumerate(learned)
        ], axis=0)

        configured_lambda = float(self.cfg.get("behavioral_adapter_lambda", 0.0))
        active_mode = str(getattr(self.env, "mode", ""))
        only_behavioral = bool(
            self.cfg.get("behavioral_adapter_only_in_behavioral_drift", True)
        )
        adapter_active = configured_lambda > 0.0 and (
            not only_behavioral or active_mode == "behavioral_drift"
        )
        lam = float(np.clip(configured_lambda, 0.0, 1.0)) if adapter_active else 0.0

        scripted = np.stack(
            [self._behavioural_execution_distribution(agent) for agent in range(self.n_agents)],
            axis=0,
        )
        scripted = np.stack([
            _masked_distribution(row, valid[agent])
            for agent, row in enumerate(scripted)
        ], axis=0)
        target_ids = self.cfg.get("behavioral_adapter_target_agents")
        target_roles = self.cfg.get("behavioral_adapter_target_roles")
        target_mask = np.zeros(self.n_agents, dtype=bool)
        if target_ids is not None:
            for agent in target_ids:
                if 0 <= int(agent) < self.n_agents:
                    target_mask[int(agent)] = True
        elif target_roles is not None:
            allowed_roles = {str(role) for role in target_roles}
            roles = getattr(self.env, "agent_role", {})
            for agent in range(self.n_agents):
                try:
                    target_mask[agent] = str(roles[agent]) in allowed_roles
                except (KeyError, IndexError, TypeError):
                    target_mask[agent] = False
        else:
            # Existing non-H2 callers retain their explicit all-agent opt-in.
            # The H2 runner always supplies a non-ego role subset.
            target_mask[:] = True

        h1_mode = str(self.cfg.get("h1_target_policy_mode", "learned"))
        h1_active = h1_mode == "scripted_uniform_mixture"
        if h1_mode not in {"learned", "scripted_uniform_mixture"}:
            raise ValueError(f"unknown h1_target_policy_mode: {h1_mode}")
        h1_uniform_mass = float(np.clip(
            self.cfg.get("h1_eval_uniform_mass", 0.10), 0.0, 1.0
        ))
        uniform = np.stack([
            _masked_distribution(valid[agent].astype(np.float32), valid[agent])
            for agent in range(self.n_agents)
        ], axis=0)

        # This is an H1 estimand policy, not an H2 manipulation: it becomes
        # pi before epsilon forcing, while the exact logged behaviour remains
        # b=(1-eps)pi+eps*q. It deliberately affects every target agent.
        executed = (
            (1.0 - h1_uniform_mass) * scripted + h1_uniform_mass * uniform
            if h1_active else learned.copy()
        )
        if adapter_active and not h1_active:
            executed[target_mask] = (
                (1.0 - lam) * learned[target_mask]
                + lam * scripted[target_mask]
            )
        executed = np.stack([
            _masked_distribution(row, valid[agent])
            for agent, row in enumerate(executed)
        ], axis=0)

        # Population means make the manipulation auditable without retaining
        # raw policy tensors in every H2 summary row.
        eps = 1e-12
        kl = np.sum(
            executed * (np.log(np.clip(executed, eps, 1.0))
                        - np.log(np.clip(learned, eps, 1.0))),
            axis=1,
        )
        tv = 0.5 * np.abs(executed - learned).sum(axis=1)
        selected = target_mask if adapter_active else np.zeros(self.n_agents, dtype=bool)
        unselected = ~selected
        diagnostics = {
            "behavioral_adapter_active": int(adapter_active),
            "behavioral_adapter_lambda": lam,
            "behavioral_adapter_kl": float(np.mean(kl)),
            "behavioral_adapter_tv": float(np.mean(tv)),
            "behavioral_adapter_target_count": int(selected.sum()),
            "behavioral_adapter_non_target_count": int(unselected.sum()),
            "behavioral_adapter_target_tv": float(np.mean(tv[selected])) if selected.any() else 0.0,
            "behavioral_adapter_non_target_tv": float(np.mean(tv[unselected])) if unselected.any() else 0.0,
            "behavioral_adapter_scripted_mass": float(
                np.mean(np.sum(executed * scripted, axis=1))
            ),
            "h1_target_policy_active": int(h1_active),
            "h1_target_policy_uniform_mass": h1_uniform_mass if h1_active else 0.0,
            "h1_target_policy_probs": executed.astype(np.float32),
        }
        return executed.astype(np.float32), scripted.astype(np.float32), diagnostics

    def _select_actions_population(
        self, obs_all, *, apply_forcing=True, collect_training_cache=True,
        force_candidate_refresh=False,
    ):
        """
        Batched policy forward for the full population.

        Semantics remain unchanged: every ego has distinct B_i, Z_i, and M_i,
        while excluding-j context remains cached by ego/j so proxy samples
        match action-time context.
        """
        # The dynamic candidate rule is evaluated before every pair-specific
        # measurement or representation update for the current environment state.
        self._refresh_dynamic_candidates(force=bool(force_candidate_refresh))
        actions = {}
        # Core latents, geometry, last actions, beliefs, and influence
        # signatures all change over time. Set-membership-only cache keys are
        # therefore insufficient across action-selection calls.
        self._reset_exclusion_caches()

        cache = {
            "belief_items_cache": {},
            "belief_summary_cache": {},
            "core_summary_cache": {},
            "periph_inputs_cache": {},
            "periph_summary_cache": {},
            "core_context_excluding": {},
            "periph_context_excluding": {},
            "proxy_context_excluding": {},
            "proxy_context_blocks": {},
            "pair_signal_cache": {},
            "value_cache": {},
            "policy_logits": None,
            # [FIX-X1] Snapshot geometry at this timestep so replay_builder can
            # construct x_ij at sample creation time. It cannot be reconstructed
            # after the episode because env.positions then contains final state.
            # Cost is negligible: 24 coordinates and 24 zone IDs per timestep.
            "geom_snapshot": (
                self.env_adapter.feature_snapshot()
                if bool(collect_training_cache) else None
            ),
            "candidate_epoch": int(self.candidate_epoch),
            "candidate_map_hash": str(self.candidate_map_hash),
            "candidate_neighbors_by_ego": {
                int(ego): tuple(int(j) for j in ids)
                for ego, ids in self.candidate_neighbors_by_ego.items()
            },
        }

        obs_batch = []
        core_batch = []
        periph_batch = []
        belief_batch = []

        for ego in range(self.n_agents):
            belief_items = self._build_belief_items_for_ego(ego)
            pair_signals = {}
            for j in self._candidate_ids(ego):
                pair = (int(ego), int(j))
                profile_timestamp = self._profile_update_step.get(pair)
                valid_action_count = int(
                    np.count_nonzero(self.env_adapter.valid_action_mask(int(j)))
                )
                pair_signals[int(j)] = self.sig_tracker.get_pair_signal(
                    ego,
                    int(j),
                    timestamp=(
                        -1 if profile_timestamp is None else int(profile_timestamp)
                    ),
                    structure_regime_id=self._current_structure_regime_id(),
                    valid_action_count=valid_action_count,
                    # A valid-action set alone establishes overlap, not a
                    # measured causal profile.  Before the first proxy score,
                    # the typed object must remain explicitly unsupported.
                    support_valid=bool(
                        profile_timestamp is not None and valid_action_count >= 2
                    ),
                )
            periph_inputs = self._build_periph_inputs_for_ego(
                ego, causal_pair_signals=pair_signals
            )

            belief_summary_np = self._belief_summary_np_from_items(belief_items)
            periph_summary_np = self._periph_summary_np_from_inputs(periph_inputs)
            core_summary_np = self._core_summary_for_ego(ego)

            cache["belief_items_cache"][ego] = belief_items
            cache["belief_summary_cache"][ego] = belief_summary_np
            cache["core_summary_cache"][ego] = core_summary_np
            cache["periph_inputs_cache"][ego] = periph_inputs
            cache["periph_summary_cache"][ego] = periph_summary_np

            cache["core_context_excluding"][ego] = {}
            cache["periph_context_excluding"][ego] = {}
            cache["proxy_context_excluding"][ego] = {}
            cache["pair_signal_cache"][ego] = pair_signals

            if bool(collect_training_cache):
                for j in self._candidate_ids(ego):
                    cache["core_context_excluding"][ego][j] = self._core_context_excluding(
                        ego, j,
                    )
                    cache["periph_context_excluding"][ego][j] = self._periph_context_excluding(
                        ego, j,
                    )

            obs_batch.append(self.env_adapter.observation(obs_all, ego))
            core_batch.append(core_summary_np)
            periph_batch.append(periph_summary_np)
            belief_batch.append(belief_summary_np)

        obs_t = torch.tensor(
            np.stack(obs_batch),
            dtype=torch.float32,
            device=self.device,
        )

        core_t = torch.tensor(
            np.stack(core_batch),
            dtype=torch.float32,
            device=self.device,
        )

        periph_t = torch.tensor(
            np.stack(periph_batch),
            dtype=torch.float32,
            device=self.device,
        )

        belief_t = torch.tensor(
            np.stack(belief_batch),
            dtype=torch.float32,
            device=self.device,
        )

        valid_action_masks = np.stack([
            self.env_adapter.valid_action_mask(agent)
            for agent in range(self.n_agents)
        ], axis=0)

        with torch.no_grad():
            logits, values = self.policy_value(
                obs_t,
                core_t,
                periph_t,
                belief_t,
                valid_action_mask=valid_action_masks,
            )

            probs = torch.softmax(logits, dim=-1)
            # One CPU/GPU synchronization for all values replaces the old
            # n_agents per-step .item() synchronizations.
            values_np = values.detach().cpu().numpy()
            learned_probs_np = probs.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()

        probs_np, scripted_probs_np, adapter_diagnostics = self._execution_policy_adapter(
            learned_probs_np,
            valid_action_masks=valid_action_masks,
        )

        if apply_forcing:
            # Sample from the actually executed pre-forcing policy. Sampling from
            # the learned distribution here would leave H2's scripted mixture as
            # a logging-only value and reintroduce the behavioural-drift no-op.
            sampled = np.asarray([
                np.random.choice(self.action_dim, p=probs_np[ego])
                for ego in range(self.n_agents)
            ], dtype=np.int64)
            actions_list = [int(sampled[ego]) for ego in range(self.n_agents)]
            pre_forcing_actions = list(actions_list)

            # [EpsilonForcedActionController] This must run after action sampling
            # and before env.step(). apply() mutates forced entries in place and
            # returns exact executed behaviour propensities.
            forced_mask, effective_probs = self.forcer.apply(
                actions=actions_list,
                policy_probs=probs_np,
                valid_action_masks=valid_action_masks,
            )
        else:
            # Evaluation-only deterministic inference path.  It deliberately
            # bypasses both categorical sampling and epsilon forcing so decision
            # fidelity / latency probes measure the architecture rather than
            # intervention noise.  Training callers retain apply_forcing=True.
            actions_list = np.argmax(probs_np, axis=-1).astype(np.int64).tolist()
            pre_forcing_actions = list(actions_list)
            forced_mask = np.zeros(self.n_agents, dtype=bool)
            effective_probs = np.asarray(probs_np, dtype=np.float64).copy()

        # ------------------------------------------------------------------
        # [VERIFY-F1] Verify that forced actions survive into the trajectory.
        #
        # The forced-action histogram must be uniform over each state's valid
        # action set. A severe deficit on an action that is valid throughout a
        # diagnostic run indicates that replay stored the pre-override action
        # or applied the override after append.
        #
        # Code order appears correct because apply mutates before packing, but
        # inspection alone cannot establish this runtime property. Measure
        # n_forced_seen, n_actually_changed relative to pre-forcing actions,
        # and hist_forced across forced agents. Correct forcing yields an
        # approximately uniform histogram. Positive n_forced_seen with zero
        # actual changes signals a defect.
        # ------------------------------------------------------------------
        try:
            fidx = (
                [k for k in range(self.n_agents) if bool(forced_mask[k])]
                if apply_forcing else []
            )
            if fidx:
                self._vf1_n_forced = getattr(self, "_vf1_n_forced", 0) + len(fidx)
                self._vf1_n_changed = getattr(self, "_vf1_n_changed", 0) + sum(
                    1 for k in fidx if actions_list[k] != pre_forcing_actions[k]
                )
                hist = getattr(self, "_vf1_hist", None)
                if hist is None:
                    hist = np.zeros(int(self.action_dim), dtype=np.int64)
                for k in fidx:
                    a = int(actions_list[k])
                    if 0 <= a < hist.shape[0]:
                        hist[a] += 1
                self._vf1_hist = hist
        except Exception:
            pass

        for ego in range(self.n_agents):
            actions[ego] = int(actions_list[ego])
            cache["value_cache"][ego] = float(values_np[ego])

        # Causal replay contexts are training-cache state and are deliberately
        # excluded from deterministic policy-inference latency probes.
        cache["proxy_context_blocks"] = {}
        if bool(collect_training_cache):
            for ego in range(self.n_agents):
                block = self._raw_proxy_context_block(ego, current_actions=actions_list)
                cache["proxy_context_blocks"][ego] = block
                for j in self._candidate_ids(ego):
                    cache["proxy_context_excluding"][ego][j] = (
                        self._raw_proxy_context_excluding_from_block(block, j)
                    )

        cache["forced_mask"] = forced_mask
        cache["action_execution_records"] = (
            tuple(self.forcer.last_execution_records) if apply_forcing else tuple()
        )
        # Paper B Eq. (13) models the pre-forcing behavioural policy action.
        # Executed actions remain separately stored for environment dynamics
        # and causal-response replay.
        cache["pre_forcing_actions"] = list(pre_forcing_actions)
        cache["behaviour_probs"] = effective_probs
        # Save raw policy probabilities (π_j(a|s)) as well; useful for
        # downstream diagnostics and for runners that expect policy_probs
        # in the trajectory. Shape: [n_agents, A]
        cache["policy_probs"] = probs_np
        cache["h1_target_policy_probs"] = adapter_diagnostics.pop(
            "h1_target_policy_probs", probs_np
        )
        cache["learned_policy_probs"] = learned_probs_np
        cache["scripted_policy_probs"] = scripted_probs_np
        cache["policy_logits"] = logits_np
        cache["behavioral_adapter"] = adapter_diagnostics
        cache["valid_action_masks"] = valid_action_masks

        return actions, cache

    # ============================================================
    # Rollout collection
    # ============================================================
    def push_proxy_replay(self, trajectory):
        """
        Push population-wide trajectory into local counterfactual proxy replay.

        This must run every episode, including Stage 0, because Stage 0 collects
        supervised proxy data even before learned structural belief takes over.
        """
        return self.replay_builder.push_trajectory_to_proxy(
            trajectory=trajectory,
            proxy_ensemble=self.proxy,
            env=self.env,
        )




    def collect_episode(self):
        """
        Collect one episode.

        Critical timing fix:
        - obs_all and actions_list are context at t.
        - env.step(actions_list) moves env to t+1.
        - pair_rel_module.step_population(obs_all, actions_list, env) must see
          geometry at t, not geometry at t+1.
        - Therefore env_snapshot_before_step is restored before step_population(),
          then restore env_snapshot_after_step after it.

        Behavioural cloning fix:
        - h_prev_snapshot is captured before step_population().
        - add_bc_transition() receives h_prev_snapshot from the previous timestep.
        - This preserves meaning:
              z_ij at context t -> predict a_j at t+1.
        """
        obs_all = self.env_adapter.reset()

        if (
            self.scheduler.in_warmup()
            and str(self.cfg.get("core_selection_mode", "structural_capacity")).strip().lower() != "full_explicit"
        ):
            for ego in range(self.n_agents):
                prior_scores = self._compute_weak_prior_scores(ego)
                self.belief_modules[ego].initialize_from_weak_prior(prior_scores)

        done = False
        trajectory = []
        ep_reward = np.zeros(self.n_agents, dtype=np.float32)

        t0 = time.time()

        prev_obs_all = None
        prev_actions = None
        prev_env_snapshot_before_step = None
        prev_h_snapshot = None
        prev_s_snapshot = None
        prev_forced_mask = None
        prev_candidate_neighbors_by_ego = None

        while not done:
            actions_dict, cache = self._select_actions_population(obs_all)
            actions_list = [actions_dict[a] for a in range(self.n_agents)]

            env_snapshot_before_step = self.env_adapter.clone_state()
            h_snapshot_before_latent_update = self.pair_rel_module.clone_full_states_np()
            s_snapshot_before_latent_update = self.pair_rel_module.clone_shadow_states_np()

            next_obs_all, rewards, done, info = self.env_adapter.step(actions_list)
            env_snapshot_after_step = self.env_adapter.clone_state()

            rewards = np.array(rewards, dtype=np.float32)
            ep_reward += rewards

            trajectory.append(
                {
                    "obs_all": [x.copy() for x in obs_all],
                    "actions": list(actions_list),
                    "pre_forcing_actions": list(cache["pre_forcing_actions"]),
                    "rewards": list(rewards),
                    "belief_items_cache": cache["belief_items_cache"],
                    "behaviour_probs": cache.get("behaviour_probs"),
                    "policy_probs": cache.get("policy_probs"),
                    "h1_target_policy_probs": cache.get("h1_target_policy_probs"),
                    "learned_policy_probs": cache.get("learned_policy_probs"),
                    "scripted_policy_probs": cache.get("scripted_policy_probs"),
                    "behavioral_adapter": cache.get("behavioral_adapter", {}),
                    "valid_action_masks": cache.get("valid_action_masks"),
                    "belief_summary_cache": cache["belief_summary_cache"],
                    "core_summary_cache": cache["core_summary_cache"],
                    "periph_inputs_cache": cache["periph_inputs_cache"],
                    "periph_summary_cache": cache["periph_summary_cache"],
                    "core_context_excluding": cache["core_context_excluding"],
                    "periph_context_excluding": cache["periph_context_excluding"],
                    "proxy_context_excluding": cache["proxy_context_excluding"],
                    "proxy_context_blocks": cache["proxy_context_blocks"],
                    "value_cache": cache["value_cache"],
                    "geom_snapshot": cache["geom_snapshot"],   # [FIX-X1]
                    "forced_mask": cache["forced_mask"],
                    "action_execution_records": cache.get("action_execution_records", ()),
                    "pair_signal_cache": cache.get("pair_signal_cache", {}),
                    "structure_regime_id": self._current_structure_regime_id(),
                    "policy_version": int(self._policy_version),
                    "candidate_epoch": int(cache.get("candidate_epoch", 0)),
                    "candidate_map_hash": str(cache.get("candidate_map_hash", "")),
                    "candidate_neighbors_by_ego": cache.get(
                        "candidate_neighbors_by_ego", {}
                    ),
                    
                    "env_snapshot_before_step": env_snapshot_before_step,
                    "env_snapshot_after_step": env_snapshot_after_step,
                    "h_snapshot_before_latent_update": h_snapshot_before_latent_update,
                    "s_snapshot_before_latent_update": s_snapshot_before_latent_update,
                    "info": info,
                    "terminated": bool(
                        isinstance(info, dict) and info.get("terminated", False)
                    ),
                    "truncated": bool(
                        isinstance(info, dict) and info.get("truncated", False)
                    ),
                }
            )

            representation_frozen = bool(
                self.cfg.get("freeze_representation_state", False)
            )
            if (
                not representation_frozen
                and
                prev_obs_all is not None
                and prev_actions is not None
                and prev_env_snapshot_before_step is not None
            ):
                self.env_adapter.restore_state(prev_env_snapshot_before_step)

                self.pair_rel_module.add_bc_transition(
                    observations={a: prev_obs_all[a] for a in range(self.n_agents)},
                    actions={a: prev_actions[a] for a in range(self.n_agents)},
                    next_actions={
                        a: cache["pre_forcing_actions"][a]
                        for a in range(self.n_agents)
                    },
                    next_forced_mask=cache["forced_mask"],
                    next_valid_action_masks=cache.get("valid_action_masks"),
                    next_actions_are_pre_forcing=True,
                    env=self.env,
                    h_prev_snapshot=prev_h_snapshot,
                    s_prev_snapshot=prev_s_snapshot,
                    cd_target_fn=self._cd_target_for_pair,
                    max_cd_target_age=self.cfg.get(
                        "cd_target_max_age_steps",
                        self.cfg.get("cd_target_max_age_episodes", 1),
                    ),
                    candidate_neighbors_by_ego=prev_candidate_neighbors_by_ego,
                )

                self.env_adapter.restore_state(env_snapshot_after_step)

                # [ReciprocityTracker] Diagnostic only; never feed this into
                # action selection. Use z_ij/s_ij at t, the same context as
                # add_bc_transition, true a_j at t+1, and whether a_i^t was
                # forced. Forcing makes a_i^t mechanically independent, so any
                # predictive gain for a_j^{t+1} is causal rather than shared-
                # observation confounding.
                if prev_forced_mask is not None:
                    self._record_reciprocity(
                        prev_h_snapshot=prev_h_snapshot,
                        # Behavioural-cloning/reciprocity targets refer to the
                        # neighbour policy action before the current forcing
                        # draw, never to the randomized intervention action.
                        actions_list=cache["pre_forcing_actions"],
                        prev_forced_mask=prev_forced_mask,
                    )

            self.env_adapter.restore_state(env_snapshot_before_step)

            if not representation_frozen:
                self.pair_rel_module.step_population(
                    obs_all=obs_all,
                    actions=actions_list,
                    env=self.env,
                )

            self.env_adapter.restore_state(env_snapshot_after_step)

            prev_obs_all = [x.copy() for x in obs_all]
            prev_actions = list(actions_list)
            prev_env_snapshot_before_step = env_snapshot_before_step
            prev_h_snapshot = h_snapshot_before_latent_update
            prev_s_snapshot = s_snapshot_before_latent_update
            prev_forced_mask = cache["forced_mask"]
            prev_candidate_neighbors_by_ego = {
                int(ego): tuple(int(j) for j in ids)
                for ego, ids in cache.get("candidate_neighbors_by_ego", {}).items()
            }

            self._interaction_step += 1

            obs_all = next_obs_all

        runtime = time.time() - t0

        adapter_rows = [
            step.get("behavioral_adapter", {}) for step in trajectory
            if isinstance(step, dict)
        ]
        if adapter_rows:
            self.last_behavioral_adapter_metrics = {
                key: float(np.mean([
                    float(row.get(key, 0.0)) for row in adapter_rows
                ]))
                for key in (
                    "behavioral_adapter_active",
                    "behavioral_adapter_lambda",
                    "behavioral_adapter_kl",
                    "behavioral_adapter_tv",
                    "behavioral_adapter_scripted_mass",
                    "behavioral_adapter_target_count",
                    "behavioral_adapter_non_target_count",
                    "behavioral_adapter_target_tv",
                    "behavioral_adapter_non_target_tv",
                )
            }
            expected_actions = np.mean(
                np.concatenate(
                    [np.asarray(step["behaviour_probs"], dtype=np.float64)
                     for step in trajectory], axis=0,
                ),
                axis=0,
            )
            realised = np.asarray(
                [action for step in trajectory for action in step["actions"]],
                dtype=np.int64,
            )
            counts = np.bincount(realised, minlength=self.action_dim).astype(np.float64)
            observed_actions = counts / max(1.0, float(counts.sum()))
            self.last_behavioral_adapter_metrics["behavioral_adapter_action_freq_tv"] = float(
                0.5 * np.abs(observed_actions - expected_actions).sum()
            )
        else:
            self.last_behavioral_adapter_metrics = {}

        # Advance the epsilon annealing schedule exactly once per episode; see
        # step_episode(). VERIFY-F1 reports a small conclusive per-episode set.
        if getattr(self, "_vf1_n_forced", 0) > 0:
            h = self._vf1_hist.astype(np.float64)
            h = h / max(h.sum(), 1.0)
            # The forcer is targeted: per-agent epsilon differs and smoke test
            # section 5 measured 6.1x concentration, so forced-agent identity
            # is nonuniform. Conditional on forcing, the chosen action must be
            # uniform over the current valid-action set.
            if self.cfg.get("debug_verbose", False):
                print(
                    f"[VERIFY-F1] forced_seen={self._vf1_n_forced} "
                    f"actually_changed={self._vf1_n_changed} "
                    f"({100.0*self._vf1_n_changed/max(1,self._vf1_n_forced):.1f}%; "
                    "the match rate depends on the state-specific valid-action set) "
                    f"hist_action_forced={np.round(h, 3).tolist()}"
                )
            self._vf1_n_forced = 0
            self._vf1_n_changed = 0
            self._vf1_hist = np.zeros(int(self.action_dim), dtype=np.int64)

        self.forcer.step_episode()

        if bool(self.cfg.get("confirmatory", False)) or bool(
            self.cfg.get("strict_causal_profile", False)
        ):
            legacy_seen = int(
                getattr(self.proxy, "signature_legacy_items_seen", 0)
                + getattr(self.periph_module, "signature_legacy_items_seen", 0)
            )
            if legacy_seen:
                raise RuntimeError(
                    "paper-contract CIG-AMF run observed legacy-derived C/D signatures"
                )

        return trajectory, ep_reward, runtime

    def _record_reciprocity(self, prev_h_snapshot, actions_list, prev_forced_mask):
        """
        Record reciprocity information gain g_ij for every directed pair using
        existing bc_head/shadow_to_full. This performs diagnostic no_grad
        forward only, with no additional training. Batch all
        n_agents*(n_agents-1) pairs rather than calling bc_head separately.
        """
        pairs = [
            (ego, j)
            for ego in range(self.n_agents)
            for j in self._candidate_ids(ego)
        ]

        if len(pairs) == 0:
            return

        z_list = []
        s_list = []

        for ego, j in pairs:
            key = (int(ego), int(j))
            z = (
                prev_h_snapshot.get(key)
                if prev_h_snapshot is not None
                else None
            )
            if z is None:
                z = self.pair_rel_module.get_pair_latent(ego, j)
            z_list.append(np.asarray(z, dtype=np.float32).reshape(-1))
            s_list.append(self.pair_rel_module.get_shadow_latent(ego, j))

        z_t = torch.tensor(
            np.stack(z_list, axis=0), dtype=torch.float32, device=self.device
        )
        s_t = torch.tensor(
            np.stack(s_list, axis=0), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            logits_with_ego = self.pair_rel_module.bc_head(z_t)
            logits_without_ego = self.pair_rel_module.bc_head(
                self.pair_rel_module.shadow_to_full(s_t)
            )

        logits_with_np = logits_with_ego.detach().cpu().numpy()
        logits_without_np = logits_without_ego.detach().cpu().numpy()

        for k, (ego, j) in enumerate(pairs):
            self.recip.record(
                ego_id=ego,
                neighbor_id=j,
                logits_with_ego=logits_with_np[k],
                logits_without_ego=logits_without_np[k],
                true_next_action=int(actions_list[j]),
                ego_was_forced=bool(prev_forced_mask[ego]),
            )

    # ============================================================
    # Policy update
    # ============================================================

    def _vtrace_targets(self, trajectory):
        """Compute V-trace critic targets and policy advantages from logged b."""
        length = len(trajectory)
        values = np.asarray([[float(step["value_cache"][ego]) for ego in range(self.n_agents)] for step in trajectory], dtype=np.float32)
        rewards = np.asarray([step["rewards"] for step in trajectory], dtype=np.float32)
        rho = np.ones_like(values)
        for t, step in enumerate(trajectory):
            behaviour = step.get("behaviour_probs")
            target = step.get("learned_policy_probs", step.get("policy_probs"))
            if behaviour is None or target is None:
                continue
            for ego, action in enumerate(step["actions"]):
                rho[t, ego] = float(target[ego][int(action)]) / max(float(behaviour[ego][int(action)]), 1e-8)
        rho_bar = np.minimum(rho, float(self.cfg.get("vtrace_rho_clip", 1.0)))
        c_bar = np.minimum(rho, float(self.cfg.get("vtrace_c_clip", 1.0)))
        targets, advantages = np.zeros_like(values), np.zeros_like(values)
        next_target = np.zeros((self.n_agents,), dtype=np.float32)
        next_value = np.zeros((self.n_agents,), dtype=np.float32)
        gamma = float(self.cfg["discount"])
        for t in reversed(range(length)):
            delta = rho_bar[t] * (rewards[t] + gamma * next_value - values[t])
            targets[t] = values[t] + delta + gamma * c_bar[t] * (next_target - next_value)
            advantages[t] = rho_bar[t] * (rewards[t] + gamma * next_target - values[t])
            next_target, next_value = targets[t], values[t]
        return targets, advantages

    def update_policy(self, trajectory):
        """
        [docs/CIG-AMF_training_debug_master.md sections 2.2(b,c)] Two corrections:

        (b) Advantage was unnormalized. The old `adv = ret_t - value`
            directly scaled -logp, tying gradient scale to volatile long-
            horizon reward scale. Normalize advantage to zero mean and unit
            variance per timestep batch. With 24 agents, each batch is large
            enough for a rough std estimate and avoids a second episode-wide
            forward. Keep critic target ret_t on its original scale; normalize
            only actor advantage.

        (c)/1.2 policy_loss previously combined actor, critic, and entropy and
            could not locate a failure. Return separate actor_loss,
            critic_loss, entropy, and grad_norm_preclip fields in a dictionary
            matching the debug schema.
        """
        T = len(trajectory)

        if T == 0:
            return {
                "loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0,
                "entropy": 0.0, "grad_norm_preclip": 0.0,
                "adv_mean": 0.0, "adv_std": 0.0,
            }

        returns, vtrace_advantages = self._vtrace_targets(trajectory)

        total_loss = 0.0
        total_periph_aux_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_importance_ratio = 0.0
        adv_mean_acc = []
        adv_std_acc = []
        count = 0

        for t, step in enumerate(trajectory):
            obs_batch = []
            core_batch = []
            actions_batch = []
            behaviour_prob_batch = []
            returns_batch = []
            belief_tensors = []
            periph_tensors = []

            for ego in range(self.n_agents):
                obs_i = self.env_adapter.observation(step["obs_all"], ego)

                obs_batch.append(obs_i)
                core_batch.append(step["core_summary_cache"][ego])
                actions_batch.append(int(step["actions"][ego]))
                behaviour_rows = step.get("behaviour_probs")
                behaviour_prob_batch.append(
                    1.0 if behaviour_rows is None else float(
                        np.asarray(behaviour_rows[ego], dtype=np.float32)[
                            int(step["actions"][ego])
                        ]
                    )
                )
                returns_batch.append(float(returns[t, ego]))

                belief_tensors.append(
                    self._belief_summary_tensor_from_items(
                        step["belief_items_cache"][ego]
                    )
                )

                # [FIX-CRIT-1] This block was previously dedented outside the
                # ego loop and used leaked `ego = n_agents-1`. It computed M_i
                # only for the last ego, then PolicyValueNet.forward silently
                # expanded [1,D] across all 24 agents. Consequences:
                # 1. Eq. 25 was violated because every agent shared agent 23's
                #    memory instead of an ego-specific M_i.
                # 2. slot_usage_ema updated from one ego per timestep, yielding
                #    poor routing diversity and entropy 0.44<0.5 despite valid
                #    semantic slots.
                # 3. Eq. 26-27 aux_loss had 1/24 expected magnitude from one ego,
                #    making No-AuxLoss nearly byte-identical to Full-CIGAMF.
                periph_out = self._periph_full_from_inputs(
                    step["periph_inputs_cache"][ego]
                )
                periph_tensors.append(periph_out["memory"])
                total_periph_aux_loss = total_periph_aux_loss + periph_out["aux_loss"]

            obs_t = torch.tensor(
                np.stack(obs_batch),
                dtype=torch.float32,
                device=self.device,
            )

            core_t = torch.tensor(
                np.stack(core_batch),
                dtype=torch.float32,
                device=self.device,
            )

            belief_t = torch.stack(belief_tensors, dim=0)
            periph_t = torch.stack(periph_tensors, dim=0)

            logits, value = self.policy_value(
                obs_t,
                core_t,
                periph_t,
                belief_t,
                valid_action_mask=step.get("valid_action_masks"),
            )
            valid_rows = step.get("valid_action_masks")
            if valid_rows is not None:
                valid_t = torch.tensor(
                    np.asarray(valid_rows, dtype=bool),
                    dtype=torch.bool,
                    device=logits.device,
                )
                if valid_t.shape != logits.shape or not bool(valid_t.any(dim=1).all()):
                    raise ValueError("trajectory valid-action masks are malformed")
                logits = logits.masked_fill(~valid_t, -torch.inf)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            action_t = torch.tensor(
                actions_batch,
                dtype=torch.long,
                device=self.device,
            )

            ret_t = torch.tensor(
                returns_batch,
                dtype=torch.float32,
                device=self.device,
            )

            logp = dist.log_prob(action_t)
            adv_raw = torch.tensor(vtrace_advantages[t], dtype=torch.float32, device=self.device).detach()

            adv_mean_acc.append(float(adv_raw.mean().item()))
            adv_std_acc.append(float(adv_raw.std(unbiased=False).item()))

            adv = (adv_raw - adv_raw.mean()) / (adv_raw.std(unbiased=False) + 1e-8)

            # Actions are collected from the known mixture behaviour policy
            # b, while this network represents pi.  Use a clipped exact
            # importance ratio rather than treating forced actions as on-policy.
            pi_obs = torch.gather(probs, 1, action_t.unsqueeze(1)).squeeze(1)
            b_obs = torch.tensor(
                behaviour_prob_batch, dtype=torch.float32, device=self.device
            )
            rho = torch.clamp(
                pi_obs.detach() / torch.clamp(b_obs, min=1e-8),
                min=0.0, max=max(1.0, float(self.cfg.get("actor_importance_clip", 2.0))),
            )
            # V-trace has already applied the clipped rho to the policy
            # advantage; multiplying it again would double-correct forcing.
            policy_loss = -logp * adv
            value_loss = F.mse_loss(value, ret_t, reduction="none")
            entropy = dist.entropy()

            critic_coeff = float(self.cfg.get("critic_loss_coeff", 0.5))
            entropy_coeff = float(self.cfg.get("entropy_coeff", 0.01))
            loss_vec = policy_loss + critic_coeff * value_loss - entropy_coeff * entropy

            total_loss = total_loss + loss_vec.sum()
            total_actor_loss = total_actor_loss + policy_loss.sum()
            total_critic_loss = total_critic_loss + value_loss.sum()
            total_entropy = total_entropy + entropy.sum()
            total_importance_ratio = total_importance_ratio + rho.sum()
            count += self.n_agents

        self.policy_optim.zero_grad()

        # [FIX-8] After FIX-CRIT-1, aux_loss is added 24 rather than once per
        # timestep, raising effective L_aux weight 24x relative to the scale
        # used to tune lambda_lb=1.2. Evidence: policy_loss rose 0.19->4.58
        # (~24x) and episode-50 reward fell to -0.711. The current per-agent-
        # step mean matches total_loss and is the correct scale, so reduce
        # lambda via periph_lb_coeff rather than altering the denominator.
        aux_term = total_periph_aux_loss / max(1, count)
        policy_term = total_loss / max(1, count)
        loss = policy_term + aux_term
        loss.backward()

        grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
            list(self.policy_value.parameters())
            + list(self.periph_module.parameters())
            + list(self.belief_summary_builder.parameters()),
            float(self.cfg.get("policy_grad_clip", 0.5)),
        )

        self.policy_optim.step()
        self._policy_version += 1

        return {
            # [FIX-8] The old loss included auxiliary terms, so aux dominated
            # policy_loss: about 4.4 of 4.58. Keep it separate to satisfy the
            # debug rule against diagnosing aggregated values.
            "loss": float(policy_term.item()),
            "aux_loss": float(aux_term.item()) if torch.is_tensor(aux_term) else float(aux_term),
            "total_loss_with_aux": float(loss.item()),
            "actor_importance_ratio_mean": float(
                (total_importance_ratio / max(1, count)).item()
            ),
            "actor_loss": float((total_actor_loss / max(1, count)).item()),
            "critic_loss": float((total_critic_loss / max(1, count)).item()),
            "entropy": float((total_entropy / max(1, count)).item()),
            "grad_norm_preclip": float(grad_norm_preclip),
            "adv_mean": float(np.mean(adv_mean_acc)) if adv_mean_acc else 0.0,
            "adv_std": float(np.mean(adv_std_acc)) if adv_std_acc else 0.0,
        }

    # ============================================================
    # Replay / proxy / graph / belief update
    # ============================================================

    def push_trajectory_to_proxy_buffer(self, trajectory):
        """
        Collect proxy samples every episode.

        Stage 0 performs no graph update but must still collect the proxy
        buffer. Pushing only when should_update_graph() would omit warm-up data.
        """
        for timestep, step in enumerate(trajectory):
            step.setdefault("episode_id", int(self.episodes_completed))
            step.setdefault("timestep", int(timestep))
        pushed = self.replay_builder.push_trajectory_to_proxy(
            trajectory=trajectory,
            proxy_ensemble=self.proxy,
            env=self.env,
        )

        return int(pushed)

    def _score_all_pairs_and_update_beliefs(
        self,
        obs_all,
        actions,
        observed_returns=None,
        behaviour_probs=None,
        policy_probs=None,
    ):
        """
        Score every directed pair (ego,j) and update Bayes-light beliefs.

        Semantics remain unchanged: use proxy-ensemble scores with context
        Z_i^{-j}, M_i^{-j}, B_i; update beliefs followed by hysteresis; and
        warm-start z_ij from shadow when j enters the core.
        """
        # Graph scoring restores historical environment snapshots and updates
        # signatures between score points, so no exclusion context may be
        # reused from action selection or a previous historical timestep.
        self._reset_exclusion_caches()

        total_promoted = 0
        total_demoted = 0

        # Structural-capacity matrix C[ego,j].  D is intentionally excluded:
        # behaviour changes should not by themselves trigger structural reset.
        influence_matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
        direction_matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
        allocation_mode = str(
            self.cfg.get("core_selection_mode", "structural_capacity")
        ).strip().lower()
        external_modes = {
            "behavioral_direction",
            "observational_correlation",
            "attention",
            "weak_prior",
            "random",
            "oracle_capacity",
            "full_explicit",
        }
        if allocation_mode not in {"structural_capacity", *external_modes}:
            raise ValueError(f"unknown core_selection_mode {allocation_mode!r}")
        association_matrix = (
            self._observational_association_matrix()
            if allocation_mode == "observational_correlation"
            else None
        )

        for ego in range(self.n_agents):
            obs_i = self.env_adapter.observation(obs_all, ego)
            action_i = int(actions[ego])

            belief_items = self._build_belief_items_for_ego(ego)
            belief_summary = self._belief_summary_np_from_items(belief_items)

            obs_i_batch = []
            action_i_batch = []
            action_j_batch = []
            z_batch = []
            m_batch = []
            b_batch = []
            context_block = self._raw_proxy_context_block(
                ego, current_actions=actions
            )
            neighbor_ids = []

            for j in self._candidate_ids(ego):

                obs_i_batch.append(obs_i)
                action_i_batch.append(action_i)
                action_j_batch.append(int(actions[j]))
                raw_core, raw_periph = self._raw_proxy_context_excluding_from_block(
                    context_block, j
                )
                z_batch.append(raw_core)
                m_batch.append(raw_periph)
                b_batch.append(belief_summary)
                neighbor_ids.append(j)

            # Prepare optional DR inputs: observed_returns and behaviour_probs
            observed_returns_batch = None
            behaviour_probs_obs_batch = None
            policy_probs_j_batch = None

            if observed_returns is not None:
                try:
                    r = float(observed_returns.get(ego, observed_returns[ego]))
                except Exception:
                    try:
                        r = float(observed_returns[ego])
                    except Exception:
                        r = None

                if r is not None:
                    observed_returns_batch = [r for _ in neighbor_ids]

            if behaviour_probs is not None:
                behaviour_probs_obs_batch = []
                for j in neighbor_ids:
                    try:
                        bp = float(behaviour_probs[j][int(actions[j])])
                    except Exception:
                        bp = None
                    behaviour_probs_obs_batch.append(bp)

            if policy_probs is not None:
                candidate_policy_rows = []
                policy_rows_valid = True
                for j in neighbor_ids:
                    try:
                        row = np.asarray(
                            policy_probs[j], dtype=np.float32
                        ).reshape(-1)
                        if (
                            row.shape != (self.action_dim,)
                            or not np.all(np.isfinite(row))
                        ):
                            policy_rows_valid = False
                    except Exception:
                        row = None
                        policy_rows_valid = False
                    candidate_policy_rows.append(row)
                if policy_rows_valid:
                    policy_probs_j_batch = candidate_policy_rows

            out = self.proxy.score_batch_from_context_block(
                context_block=context_block,
                target_ids=neighbor_ids,
                obs_i_batch=obs_i_batch,
                action_i_batch=action_i_batch,
                observed_action_j_batch=action_j_batch,
                z_core_excl_j_batch=z_batch,
                m_periph_excl_j_batch=m_batch,
                belief_summary_batch=b_batch,
                policy_probs_j_batch=policy_probs_j_batch,
                observed_returns_batch=observed_returns_batch,
                behaviour_probs_obs_batch=behaviour_probs_obs_batch,
                # [FIX-X1] Build x_ij from current online environment state
                # using the same function as the push path to prevent
                # train/serve skew.
                pair_feat_batch=[
                    self.env_adapter.pair_features(ego, j)
                    for j in neighbor_ids
                ],
                valid_action_mask_batch=np.stack([
                    self.env_adapter.valid_action_mask(j)
                    for j in neighbor_ids
                ], axis=0),
            )
            c_mu_arr = out.get("c_mu", out.get("mu_range"))
            c_sigma_arr = out.get("c_sigma", out["sigma"])
            d_mu_arr = out.get("d_mu", out["mu"])

            # Contextuality uses a coarse pre-treatment interaction category,
            # not target zone alone. Support validity still requires repeated
            # observations in at least two categories.
            self.sig_tracker.update_from_proxy_output(
                ego_id=ego,
                neighbor_ids=neighbor_ids,
                proxy_out=out,
                context_keys=[
                    self._signature_context_key(ego, j) for j in neighbor_ids
                ],
            )
            for j in neighbor_ids:
                self._profile_update_step[(int(ego), int(j))] = int(
                    getattr(self, "_interaction_step", 0)
                )

            mu_sigma = {
                j: (float(c_mu_arr[k]), float(c_sigma_arr[k]))
                for k, j in enumerate(neighbor_ids)
            }

            for k, j in enumerate(neighbor_ids):
                influence_matrix[ego, j] = float(c_mu_arr[k])
                direction_matrix[ego, j] = float(d_mu_arr[k])

            if allocation_mode in external_modes:
                self.belief_modules[ego].update_evidence(mu_sigma)
                update_result = (set(), set())
            else:
                update_result = self.belief_modules[ego].update_batch(mu_sigma)

            if update_result is None:
                promoted = set()
                demoted = set()
            else:
                promoted, demoted = update_result

            if allocation_mode in external_modes:
                selector_scores = self._external_selector_scores(
                    allocation_mode,
                    ego,
                    neighbor_ids,
                    d_mu_arr,
                    association_matrix,
                    belief_items=belief_items,
                )
                target_size = (
                    len(neighbor_ids)
                    if allocation_mode == "full_explicit"
                    else self.belief_modules[ego]._effective_max_k()
                )
                promoted, demoted = self.belief_modules[
                    ego
                ].select_core_from_external_scores(
                    selector_scores,
                    target_size=target_size,
                )

            total_promoted += len(promoted)
            total_demoted += len(demoted)

        self.pair_rel_module.reconcile_core_sets(
            {
                ego: module.get_core_set()
                for ego, module in self.belief_modules.items()
            },
            warm_start=bool(self.cfg.get("pair_warm_start", True)),
        )
        self._latest_direction_matrix = direction_matrix
        return int(total_promoted), int(total_demoted), influence_matrix

    def update_graph_modules(self, trajectory):
        """
        Train proxy and update belief/core.

        Replay is no longer pushed here; run() pushes it every episode. This
        method handles only the slow structural update.
        """
        if len(trajectory) == 0:
            return {
                "proxy_loss": 0.0,
                "proxy_train_residual": 0.0,
                "proxy_holdout_residual": 0.0,
                "triggered": 0,
                "promoted": 0,
                "demoted": 0,
            }
        if self.cfg.get("debug_verbose", False):
            print(f"[GRAPH-DEBUG] should_update_graph={self.scheduler.should_update_graph()} " f"buffer_len={len(self.proxy.buffer)} n_steps_cfg={self.cfg['proxy_train_steps']}")
            st = self.scheduler.get_status()
            print(f"[SCHED-DEBUG] episode={st['episode']} freq_used={self.scheduler._accel_freq() if self.scheduler.accel_remaining>0 else self.scheduler._base_freq()} " f"accel_remaining={st['accel_remaining']} trigger_count={st['trigger_count']} last_trigger_ep={st['last_trigger_episode']}")
        proxy_loss = self.proxy.train_step(
            n_steps=self.cfg["proxy_train_steps"],
            batch_size=self.cfg["proxy_batch_size"],
            holdout_size=self.cfg["proxy_holdout_size"],
        )

        score_steps = int(max(1, self.cfg.get("graph_score_steps", 1)))
        step_indices = np.linspace(
            0,
            len(trajectory) - 1,
            num=min(score_steps, len(trajectory)),
            dtype=int,
        ).tolist()

        # Precompute H-step returns for all timesteps to provide
        # observed_returns into the proxy scoring call for DR correction.
        try:
            h_returns = self.replay_builder.build_h_step_returns(
                trajectory, self.n_agents, complete_only=True
            )
        except Exception:
            h_returns = [None for _ in range(len(trajectory))]

        promoted = 0
        demoted = 0
        last_influence_matrix = None

        for idx in step_indices:
            step = trajectory[int(idx)]
            self.env_adapter.restore_state(step["env_snapshot_before_step"])
            observed_returns = h_returns[int(idx)] if h_returns is not None else None
            behaviour_probs = step.get("behaviour_probs") if isinstance(step, dict) else None
            policy_probs = step.get("policy_probs") if isinstance(step, dict) else None

            p_i, d_i, w_i = self._score_all_pairs_and_update_beliefs(
                obs_all=step["obs_all"],
                actions=step["actions"],
                observed_returns=observed_returns,
                behaviour_probs=behaviour_probs,
                policy_probs=policy_probs,
            )
            promoted += int(p_i)
            demoted += int(d_i)
            last_influence_matrix = w_i

        self.env_adapter.restore_state(trajectory[-1]["env_snapshot_after_step"])

        residual = self.proxy.get_latest_residual()

        # The frozen blinded probe is the prespecified trigger. Matrix
        # movement is retained below only as a diagnostic covariate.
        drift_info = self.drift.step(
            episode=int(self.scheduler.episode),
            buffer=self.proxy.buffer,
            n_train_batches=self.cfg.get("drift_train_batches", 5),
        )
        probe_z = float(drift_info.get("z", 0.0) or 0.0)
        probe_cusum = float(drift_info.get("cusum", 0.0) or 0.0)

        # Trigger 2: jump in the directional diagnostic matrix.
        matrix_z = 0.0
        if last_influence_matrix is not None:
            self.matdet.update(last_influence_matrix)
            matrix_z = float(self.matdet.z_score())

        monitoring_ready = bool(self.drift.is_monitoring_ready())
        if bool(self.cfg.get("disable_drift_detector", False)):
            fire_info = {
                "fired": False, "reason": "detector_disabled",
                "n_inflated": 0, "probe_z": probe_z,
                "probe_cusum": probe_cusum, "matrix_z": matrix_z,
            }
        elif not monitoring_ready:
            fire_info = {
                "fired": False, "reason": "detector_not_ready",
                "n_inflated": 0, "probe_z": 0.0,
                "probe_cusum": 0.0, "matrix_z": matrix_z,
            }
        else:
            fire_info = self.scheduler.evaluate_drift(
                probe_z=probe_z,
                probe_cusum=probe_cusum,
                matrix_z=matrix_z,
                belief_modules=self.belief_modules,
                drift_detector=self.drift,
            )
        if fire_info["fired"]:
            print(f"[DRIFT-FIRE] ep={self.scheduler.episode} reason={fire_info['reason']} n_inflated={fire_info['n_inflated']} "
                  f"probe_z={fire_info['probe_z']:.2f} "
                  f"probe_cusum={fire_info.get('probe_cusum', 0.0):.2f} "
                  f"matrix_z={fire_info['matrix_z']:.2f}")
        triggered = int(fire_info["fired"])

        return {
            "proxy_loss": float(proxy_loss),
            "proxy_train_residual": float(
                getattr(self.proxy, "get_latest_train_residual", lambda: residual)()
            ),
            "proxy_holdout_residual": float(
                getattr(self.proxy, "get_latest_holdout_residual", lambda: residual)()
            ),
            "triggered": int(triggered),
            "probe_z": float(probe_z),
            "probe_cusum": float(probe_cusum),
            "drift_phase": str(drift_info.get("phase", "unknown")),
            "drift_monitoring_ready": bool(self.drift.is_monitoring_ready()),
            "matrix_z": float(matrix_z),
            "promoted": int(promoted),
            "demoted": int(demoted),
        }

    # ============================================================
    # Metrics
    # ============================================================

    def _core_f1(self, pred_core, gt_core):
        pred = set(pred_core)
        gt = set(gt_core)

        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        return 2.0 * precision * recall / (precision + recall)

    def _belief_population_stats(self):
        capacity_vals = []
        g_vals = []

        for ego in range(self.n_agents):
            state = self.belief_modules[ego].get_state_dict()

            for _, item in state.items():
                capacity_vals.append(float(item["capacity_bar"]))
                g_vals.append(float(item.get("g_score", item.get("lcb_score", 0.0))))

        mean_capacity = float(np.mean(np.clip(capacity_vals, 0.0, None))) if capacity_vals else 0.0
        max_g = float(np.max(g_vals)) if g_vals else 0.0

        return mean_capacity, max_g

    def _causal_latency_stats(self):
        """Summarise typed Paper-A latency diagnostics without changing B's 5D view."""
        signals = []
        for ego in range(self.n_agents):
            for neighbor in self._candidate_ids(ego):
                signals.append(
                    self.sig_tracker.get_pair_signal(
                        ego,
                        neighbor,
                        timestamp=int(self._profile_update_step.get(
                            (int(ego), int(neighbor)), -1
                        )),
                        structure_regime_id=self._current_structure_regime_id(),
                        valid_action_count=int(np.count_nonzero(
                            self.env_adapter.valid_action_mask(neighbor)
                        )),
                        support_valid=bool(
                            (int(ego), int(neighbor)) in self._profile_update_step
                            and np.count_nonzero(
                                self.env_adapter.valid_action_mask(neighbor)
                            ) >= 2
                        ),
                    )
                )
        if not signals:
            return 0.0, 0.0, 0.0
        valid = [signal for signal in signals if signal.latency_valid]
        onset_valid = [signal for signal in signals if signal.latency_onset_valid]
        return (
            float(len(valid)) / float(len(signals)),
            float(len(onset_valid)) / float(len(signals)),
            float(np.mean([signal.latency_cm for signal in valid])) if valid else 0.0,
        )

    def get_influence_matrix(self):
        """Return the current structural-capacity matrix C used by the core."""
        matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
        for ego, belief in self.belief_modules.items():
            for neighbor in belief.neighbor_ids:
                matrix[int(ego), int(neighbor)] = max(
                    0.0, float(belief.debiased_mu(int(neighbor)))
                )
        return matrix

    def get_direction_matrix(self):
        """Return the latest D matrix; it is never used for core selection."""
        return np.asarray(
            getattr(
                self,
                "_latest_direction_matrix",
                np.zeros((self.n_agents, self.n_agents), dtype=np.float64),
            ),
            dtype=np.float64,
        ).copy()

    def _observational_association_matrix(self):
        """Estimate action/reward association without interventional labels."""
        grouped = {}
        discount = float(self.cfg.get("discount", 1.0))
        for sample in self.proxy.buffer:
            if not bool(sample.get("horizon_complete", True)):
                continue
            key = (int(sample["ego_id"]), int(sample["neighbor_id"]))
            lag = np.asarray(
                sample.get("target_lag_rewards", []), dtype=np.float64
            ).reshape(-1)
            if lag.size == 0:
                continue
            weights = discount ** np.arange(lag.size, dtype=np.float64)
            grouped.setdefault(key, []).append((
                int(sample["observed_action_j"]),
                float(np.dot(weights, lag)),
            ))
        matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
        min_support = int(self.cfg.get("selector_correlation_min_support", 5))
        for (ego, neighbor), records in grouped.items():
            actions = np.asarray([item[0] for item in records], dtype=np.int64)
            outcomes = np.asarray([item[1] for item in records], dtype=np.float64)
            if outcomes.size < 2 * min_support or float(np.std(outcomes)) <= 1e-12:
                continue
            best = 0.0
            for action in range(self.action_dim):
                indicator = (actions == action).astype(np.float64)
                support = int(indicator.sum())
                if support < min_support or outcomes.size - support < min_support:
                    continue
                corr = np.corrcoef(indicator, outcomes)[0, 1]
                if np.isfinite(corr):
                    best = max(best, abs(float(corr)))
            matrix[ego, neighbor] = best
        return matrix

    def _external_selector_scores(
        self, mode, ego, neighbor_ids, direction_values, association_matrix,
        belief_items=None,
    ):
        """Return scores for a non-C Paper-B selector ablation."""
        mode = str(mode).strip().lower()
        if mode == "behavioral_direction":
            return {
                int(j): abs(float(direction_values[index]))
                for index, j in enumerate(neighbor_ids)
            }
        if mode == "observational_correlation":
            return {
                int(j): float(association_matrix[int(ego), int(j)])
                for j in neighbor_ids
            }
        if mode == "attention":
            if belief_items is None:
                raise ValueError("attention selector requires current belief items")
            items = torch.as_tensor(
                np.asarray(belief_items, dtype=np.float32), device=self.device
            )
            if items.ndim != 2 or items.shape[0] != len(neighbor_ids):
                raise ValueError("attention selector belief-item alignment failed")
            with torch.no_grad():
                encoded = self.belief_summary_builder.item_enc(items)
                logits = self.belief_summary_builder.attn(encoded).squeeze(-1)
                weights = torch.softmax(logits, dim=0).cpu().numpy()
            return {
                int(j): float(weights[index])
                for index, j in enumerate(neighbor_ids)
            }
        if mode == "weak_prior":
            return {
                int(j): float(self.env_adapter.weak_prior_score(int(ego), int(j)))
                for j in neighbor_ids
            }
        if mode == "random":
            # Persist one deterministic RNG stream per ego. Re-seeding on every
            # call made the old "Random-Core" a fixed ranking across all states.
            ego = int(ego)
            rng = self._selector_random_rngs.get(ego)
            if rng is None:
                seed = int(self.cfg.get("seed", 0)) + 104729 * (ego + 1)
                rng = np.random.RandomState(seed)
                self._selector_random_rngs[ego] = rng
            return {int(j): float(rng.uniform()) for j in neighbor_ids}
        if mode == "oracle_capacity":
            table = getattr(self, "oracle_capacity_scores_by_ego", None)
            if not isinstance(table, dict) or int(ego) not in table:
                raise RuntimeError(
                    "oracle_capacity selector requires clone-state C* scores; "
                    "mechanism coefficients are not an Oracle-C substitute"
                )
            row = table[int(ego)]
            return {
                int(j): max(0.0, float(row.get(int(j), 0.0)))
                for j in neighbor_ids
            }
        if mode == "full_explicit":
            return {int(j): 1.0 for j in neighbor_ids}
        raise ValueError(f"unknown external core selector {mode!r}")

    def _signature_context_key(self, ego, neighbor):
        """Return the adapter-defined pre-treatment interaction category."""
        return self.env_adapter.context_key(int(ego), int(neighbor))

    def evaluate_episode_snapshot(
        self,
        trajectory,
        episode_reward,
        rollout_runtime,
        episode_runtime_total=None,
    ):
        last_info = trajectory[-1]["info"]

        f1s = []
        tvars = []
        uncs = []
        core_sizes = []
        core_switches = []

        diagnostic = last_info.get("diagnostic_core_by_ego", None)
        f1s_role = []   # [F1-TOPK]

        # ------------------------------------------------------------------
        # Core F1 uses a fixed oracle-capacity top-k target.  Its cardinality
        # must not depend on the predicted core size: adapting k to a model
        # output makes a recovery metric self-serving and non-comparable.
        #
        # gt_core_by_ego/diagnostic_core_by_ego are declared role lists. After
        # Phi became measured from the continuous SGTP formula, these core
        # definitions diverged: beliefs select by |mu| while the old metric
        # scored roles. F1=0.000 throughout a run was a metric defect, not a
        # model failure.
        #
        # Report both mean_f1 against measured top-k |Phi| as the primary
        # metric and mean_f1_role against role labels as evidence that roles
        # ceased to be valid ground truth after the Phi refactor.
        # ------------------------------------------------------------------
        gt_influence = last_info.get("gt_influence_by_ego", None)

        for ego in range(self.n_agents):
            pred_core = self.belief_modules[ego].get_core_set()

            gt_core_role = set() if diagnostic is None else set(diagnostic[ego])

            gt_core = gt_core_role
            if gt_influence is not None and ego in gt_influence:
                row = gt_influence[ego]
                if row:
                    k = int(max(1, self.cfg.get(
                        "ground_truth_core_k", self.cfg.get("seed_core_top_k", 3)
                    )))
                    gt_core = set(
                        sorted(row, key=lambda j: abs(float(row[j])),
                               reverse=True)[:int(k)]
                    )

            f1s.append(self._core_f1(pred_core, gt_core))
            f1s_role.append(self._core_f1(pred_core, gt_core_role))
            tvars.append(self.belief_modules[ego].get_temporal_variance())
            uncs.append(self.belief_modules[ego].get_mean_uncertainty())
            core_sizes.append(len(pred_core))

            if hasattr(self.belief_modules[ego], "get_core_switch_count"):
                core_switches.append(
                    self.belief_modules[ego].get_core_switch_count()
                )
            else:
                prev_core = getattr(self.belief_modules[ego], "prev_core_set", set())
                curr_core = getattr(self.belief_modules[ego], "core_set", set())
                core_switches.append(len(set(prev_core) ^ set(curr_core)))

        agent_steps = float(self.n_agents * len(trajectory))
        throughput = agent_steps / max(float(rollout_runtime), 1e-9)
        total_runtime = float(
            episode_runtime_total
            if episode_runtime_total is not None
            else rollout_runtime
        )
        throughput_total = agent_steps / max(total_runtime, 1e-9)

        mean_capacity, max_g = self._belief_population_stats()
        latency_valid_fraction, latency_onset_valid_fraction, latency_cm_mean = (
            self._causal_latency_stats()
        )

        # Explicit accounting required by Paper B's scaling contract.  A
        # dense run reports its true quadratic edge count; only a bounded
        # adapter candidate set can legitimately be interpreted as edge-linear.
        edge_count = int(self.measured_edge_count)
        core_pair_count = int(len(getattr(
            self.pair_rel_module, "active_core_pairs", ()
        )))
        candidate_degree_mean = float(edge_count) / float(max(1, self.n_agents))
        core_degree_mean = float(core_pair_count) / float(max(1, self.n_agents))
        n_steps = int(len(trajectory))

        def _state_bytes(states):
            total = 0
            for value in (states or {}).values():
                if isinstance(value, torch.Tensor):
                    total += int(value.numel() * value.element_size())
            return int(total)

        shadow_bytes = _state_bytes(getattr(self.pair_rel_module, "shadow_states", {}))
        core_bytes = _state_bytes(getattr(self.pair_rel_module, "full_states", {}))
        periphery_items = int(sum(
            len(self.belief_modules[ego].get_peripheral_set())
            for ego in range(self.n_agents)
        ))

        return {
            "mean_reward": float(np.mean(episode_reward)),
            "reward_per_agent": float(np.mean(episode_reward)),
            "mean_f1": float(np.mean(f1s)),
            "mean_f1_role": float(np.mean(f1s_role)) if f1s_role else 0.0,
            "mean_temporal_var": float(np.mean(tvars)),
            "mean_uncertainty": float(np.mean(uncs)),
            "mean_core_size": float(np.mean(core_sizes)),
            "mean_core_switches": float(np.mean(core_switches)),
            "mean_capacity": float(mean_capacity),
            "max_g": float(max_g),
            "causal_latency_valid_fraction": float(latency_valid_fraction),
            "causal_latency_onset_valid_fraction": float(
                latency_onset_valid_fraction
            ),
            "causal_latency_cm_mean": float(latency_cm_mean),
            # Deprecated telemetry aliases retained for historical CSV readers.
            "mean_mu": float(mean_capacity),
            "max_p": float(max_g),
            "runtime": float(rollout_runtime),
            "throughput_agent_steps_per_sec": float(throughput),
            "episode_runtime_total": float(total_runtime),
            "throughput_total_agent_steps_per_sec": float(throughput_total),
            "measured_edge_count": int(self.measured_edge_count),
            "candidate_max_degree": (
                None if self.candidate_max_degree is None
                else int(self.candidate_max_degree)
            ),
            "candidate_construction_subquadratic": bool(
                self.candidate_construction_subquadratic
            ),
            "candidate_construction_linear_candidate": bool(
                self.candidate_construction_linear_candidate
            ),
            "feature_snapshot_subquadratic": bool(self.feature_snapshot_subquadratic),
            "feature_snapshot_linear_candidate": bool(self.feature_snapshot_linear_candidate),
            "candidate_epoch": int(self.candidate_epoch),
            "candidate_map_hash": str(self.candidate_map_hash),
            "candidate_provider": str(self._candidate_telemetry.get("provider", "unknown")),
            "candidate_refresh_ms": float(
                self._candidate_telemetry.get("candidate_refresh_ms", 0.0)
            ),
            "candidate_provider_work_units": int(
                self._candidate_telemetry.get("provider_work_units", 0)
            ),
            "candidate_cell_occupancy_mean": float(
                self._candidate_telemetry.get("cell_occupancy_mean", float("nan"))
            ),
            "candidate_cell_occupancy_max": int(
                self._candidate_telemetry.get("cell_occupancy_max", -1)
            ),
            "candidate_added_pairs": int(
                self._candidate_telemetry.get("candidate_added_pairs", 0)
            ),
            "candidate_removed_pairs": int(
                self._candidate_telemetry.get("candidate_removed_pairs", 0)
            ),
            "candidate_churn": float(
                self._candidate_telemetry.get("candidate_churn", 0.0)
            ),
            "active_core_pair_count": int(
                len(getattr(self.pair_rel_module, "active_core_pairs", ()))
            ),
            "shadow_pair_count": int(len(getattr(self.pair_rel_module, "shadow_states", {}))),
            "full_state_pair_count": int(len(getattr(self.pair_rel_module, "full_states", {}))),
            "belief_pair_count": int(sum(
                len(module.neighbor_ids) for module in self.belief_modules.values()
            )),
            "signature_pair_count": int(
                self.sig_tracker.live_pair_count()
                if hasattr(self.sig_tracker, "live_pair_count") else 0
            ),
            "degree_limited_ego_count": int(sum(
                bool(getattr(module, "core_budget_degree_limited", False))
                for module in self.belief_modules.values()
            )),
            "N": int(self.n_agents),
            "E_t": edge_count,
            "K_t": core_pair_count,
            "d_bar": candidate_degree_mean,
            "k_bar": core_degree_mean,
            "proxy_pair_updates": int(edge_count * n_steps),
            "shadow_pair_updates": int(edge_count * n_steps),
            "core_pair_updates": int(core_pair_count * n_steps),
            "periphery_items_processed": periphery_items,
            "shadow_bytes": shadow_bytes,
            "core_state_bytes": core_bytes,
            "wallclock_ms": float(total_runtime * 1000.0),
        }

    # ============================================================
    # Main loop
    # ============================================================

    def _should_update_graph_this_episode(self):
        """Return the graph-update cadence selected by the scheduler."""
        # Fixed-core representation panels hold allocation constant by
        # protocol.  This is different from freezing policy learning: pair
        # shadow/full states may still evolve while the selector is held.
        if bool(self.cfg.get("freeze_graph_updates", False)):
            return False
        if bool(self.cfg.get("force_graph_update_every_episode", False)):
            return True
        return self.scheduler.should_update_graph()

    def run(self, n_episodes=100, eval_every=10):
        """
        Main training loop.

        Ordering:
        1. collect trajectory.
        2. push trajectory to proxy buffer every episode.
        3. fast update policy/value.
        4. fast update BC head.
        5. if should_update_graph(): train proxy + update belief/core.
        6. evaluate.
        7. scheduler.step_episode().
        8. if Stage 0 -> Stage 1 transition happened, reset switch counters.
        """
        n_episodes = int(n_episodes)
        eval_every = int(eval_every)
        if n_episodes < 0:
            raise ValueError("n_episodes must be non-negative")
        if eval_every <= 0:
            raise ValueError("eval_every must be positive")

        for local_ep in range(n_episodes):
            # One-based global numbering preserves the environment's exact
            # reset semantics: behavioral phase 1 begins in episode 40, while
            # the structural relocation after 40 completed episodes is logged
            # in episode 41.
            episode_number = int(self.episodes_completed) + 1
            stage_before_episode_step = int(self.scheduler.stage)

            trajectory, episode_reward, runtime = self.collect_episode()

            pushed_proxy_samples = self.push_trajectory_to_proxy_buffer(trajectory)

            t_policy = time.time()
            learning_frozen = bool(self.cfg.get("freeze_policy_learning", False))
            policy_update_info = (
                {
                    "loss": 0.0,
                    "aux_loss": 0.0,
                    "actor_loss": 0.0,
                    "critic_loss": 0.0,
                    "entropy": 0.0,
                    "grad_norm_preclip": 0.0,
                    "adv_mean": 0.0,
                    "adv_std": 0.0,
                }
                if learning_frozen
                else self.update_policy(trajectory)
            )
            policy_loss = policy_update_info["loss"]
            policy_runtime = time.time() - t_policy

            t_bc = time.time()
            bc_learning_frozen = bool(
                self.cfg.get("freeze_pair_bc_learning", learning_frozen)
            )
            bc_loss = 0.0 if bc_learning_frozen else self.pair_rel_module.train_bc(
                n_steps=self.cfg["bc_train_steps"],
                batch_size=self.cfg["bc_batch_size"],
                heads=self.heads,
                heads_optim=self.heads_optim,
                w_contrastive=self.cfg.get("heads_w_contrastive", 0.3),
                w_influence=self.cfg.get("heads_w_influence", 1.0),
            )
            bc_runtime = time.time() - t_bc

            graph_runtime = 0.0

            graph_info = {
                "proxy_loss": 0.0,
                "proxy_train_residual": 0.0,
                "proxy_holdout_residual": 0.0,
                "triggered": 0,
                "promoted": 0,
                "demoted": 0,
            }

            if self._should_update_graph_this_episode():
                t_graph = time.time()
                graph_info = self.update_graph_modules(trajectory)
                graph_runtime = time.time() - t_graph

            episode_runtime_total = runtime + policy_runtime + bc_runtime + graph_runtime

            snapshot = self.evaluate_episode_snapshot(
                trajectory,
                episode_reward,
                rollout_runtime=runtime,
                episode_runtime_total=episode_runtime_total,
            )

            stage_now = int(self.scheduler.stage)
            trigger_count_now = int(self.scheduler.trigger_count)

            self.scheduler.step_episode()

            stage_after_episode_step = int(self.scheduler.stage)

            # Semantic thresholds may be calibrated after warm-up because D's
            # reward scale is environment-dependent.  Structural C selection
            # remains fixed: reactive tau/sigma/kappa changes would turn the
            # stated uncertainty-penalized rule into a hidden threshold controller.
            is_stage_transition = (stage_before_episode_step == 0 and stage_after_episode_step == 1)
            semantic_router_frozen = bool(self.cfg.get("confirmatory", False)) or bool(
                self.cfg.get("semantic_router_frozen", False)
            )
            is_periodic_calib = (
                not semantic_router_frozen
                and
                stage_after_episode_step == 1
                and (
                    episode_number % int(self.cfg.get("semantic_calibration_every", 25)) == 0
                )
            )

            if is_stage_transition:
                self._reset_switch_counters_if_available()
                cd_norm_fitted = self.pair_rel_module.fit_cd_normalization(
                    min_samples=self.cfg.get("cd_normalization_min_samples", 32)
                )
                if bool(self.cfg.get("confirmatory", False)) and not cd_norm_fitted:
                    raise RuntimeError(
                        "confirmatory run could not freeze C/D normalization "
                        "from the pre-confirmatory training replay"
                    )

            # Validation may fit router thresholds from the development
            # stream.  Confirmatory runs use the explicit frozen values
            # supplied in config and never retune them on their own results.
            if (is_stage_transition or is_periodic_calib) and not semantic_router_frozen:
                calib_reason = "stage transition" if is_stage_transition else "periodic"
                print(f"[SEMANTIC-CALIBRATION] ep={episode_number} reason={calib_reason}")

                try:
                    calib = self.sig_tracker.auto_calibrate()
                    tau_role = calib["tau_role"]
                    sigma_hi = calib["sigma_hi"]

                    if hasattr(self, 'periph_module'):
                        self.periph_module.set_role_thresholds(
                            tau_role, sigma_hi,
                        )
                    print(
                        f"[SEMANTIC-CALIBRATION] tau_D={tau_role:.6f} "
                        f"sigma_D_hi={sigma_hi:.6f}; C priority rule unchanged"
                    )
                except Exception as e:
                    print(f"[SEMANTIC-CALIBRATION][ERROR] {type(e).__name__}: {e}")

            last_info = trajectory[-1].get("info", {}) if trajectory else {}
            structural_shift_magnitude = float(
                last_info.get("delta_phi_frobenius_structural", 0.0) or 0.0
            )
            controlled_structural_shift = int(
                last_info.get("controlled_structural_shift", 0) or 0
            )
            behavioral_phase = _current_behavioral_phase(getattr(self, "env", None))
            behavioral_shift = int(
                last_info.get("controlled_behavioral_shift", 0) or
                getattr(self, "_last_behavioral_phase", None) is not None
                and behavioral_phase != self._last_behavioral_phase
            )
            self._last_behavioral_phase = behavioral_phase
            adapter_metrics = getattr(self, "last_behavioral_adapter_metrics", {})
            self.episode_events.append({
                "episode": episode_number,
                "triggered": int(graph_info["triggered"]),
                "trigger_count": int(trigger_count_now),
                "drift_phase": str(graph_info.get("drift_phase", "unknown")),
                "drift_monitoring_ready": int(
                    bool(graph_info.get("drift_monitoring_ready", False))
                ),
                "structural_shift": int(
                    controlled_structural_shift
                    or structural_shift_magnitude > 0.0
                ),
                "structural_shift_magnitude": structural_shift_magnitude,
                "behavioral_phase": behavioral_phase,
                "behavioral_shift": behavioral_shift,
                "mean_f1": float(snapshot["mean_f1"]),
                "mean_reward": float(snapshot["mean_reward"]),
                "behavioral_adapter_active": int(
                    round(float(adapter_metrics.get("behavioral_adapter_active", 0.0)))
                ),
                "behavioral_adapter_lambda": float(
                    adapter_metrics.get("behavioral_adapter_lambda", 0.0)
                ),
                "behavioral_adapter_kl": float(
                    adapter_metrics.get("behavioral_adapter_kl", 0.0)
                ),
                "behavioral_adapter_tv": float(
                    adapter_metrics.get("behavioral_adapter_tv", 0.0)
                ),
                "behavioral_adapter_action_freq_tv": float(
                    adapter_metrics.get("behavioral_adapter_action_freq_tv", 0.0)
                ),
                "behavioral_adapter_target_count": int(
                    round(float(adapter_metrics.get("behavioral_adapter_target_count", 0.0)))
                ),
                "behavioral_adapter_non_target_count": int(
                    round(float(adapter_metrics.get("behavioral_adapter_non_target_count", 0.0)))
                ),
                "behavioral_adapter_target_tv": float(
                    adapter_metrics.get("behavioral_adapter_target_tv", 0.0)
                ),
                "behavioral_adapter_non_target_tv": float(
                    adapter_metrics.get("behavioral_adapter_non_target_tv", 0.0)
                ),
            })
            self.episodes_completed = episode_number

            # Always retain the terminal point of a call.  This preserves an
            # exact chunk-boundary snapshot without resetting the global
            # evaluation cadence when ``run`` is called repeatedly.
            should_record = (
                episode_number % eval_every == 0
                or local_ep == n_episodes - 1
            )
            if should_record:
                self.history["episodes"].append(episode_number)
                self.history["mean_reward"].append(snapshot["mean_reward"])
                self.history["reward_per_agent"].append(snapshot["reward_per_agent"])
                self.history["mean_f1"].append(snapshot["mean_f1"])
                self.history["mean_f1_role"].append(
                    snapshot.get("mean_f1_role", 0.0))
                self.history["mean_temporal_var"].append(snapshot["mean_temporal_var"])
                self.history["mean_uncertainty"].append(snapshot["mean_uncertainty"])
                self.history["mean_core_size"].append(snapshot["mean_core_size"])
                self.history["mean_core_switches"].append(snapshot["mean_core_switches"])
                self.history["mean_mu"].append(snapshot["mean_mu"])
                self.history["max_p"].append(snapshot["max_p"])
                self.history.setdefault("mean_capacity", []).append(
                    snapshot.get("mean_capacity", snapshot.get("mean_mu", 0.0))
                )
                self.history.setdefault("max_g", []).append(
                    snapshot.get("max_g", snapshot.get("max_p", 0.0))
                )
                self.history["runtime"].append(snapshot["runtime"])
                self.history["throughput_agent_steps_per_sec"].append(
                    snapshot["throughput_agent_steps_per_sec"]
                )
                self.history["episode_runtime_total"].append(
                    snapshot["episode_runtime_total"]
                )
                self.history["throughput_total_agent_steps_per_sec"].append(
                    snapshot["throughput_total_agent_steps_per_sec"]
                )
                self.history["proxy_train_residual"].append(
                    float(graph_info.get("proxy_train_residual", 0.0))
                )
                self.history["proxy_holdout_residual"].append(
                    float(graph_info.get("proxy_holdout_residual", 0.0))
                )
                sched_status = self.scheduler.get_status()
                # Preserve the legacy storage keys while retaining their
                # literal Page-CUSUM semantics: raw frozen-probe z and the
                # accumulated calibrated statistic. Matrix movement remains
                # a separate diagnostic and never substitutes for CUSUM.
                self.history["scheduler_residual_ewma"].append(
                    float(graph_info.get("probe_z", 0.0) or 0.0)
                )
                self.history["scheduler_cusum_score"].append(
                    float(graph_info.get("probe_cusum", 0.0) or 0.0)
                )
                self.history["drift_monitoring_ready"].append(
                    int(bool(graph_info.get("drift_monitoring_ready", False)))
                )
                self.history["drift_phase"].append(
                    str(graph_info.get("drift_phase", "unknown"))
                )
                self.history["scheduler_accel_remaining"].append(
                    int(sched_status.get("accel_remaining", 0))
                )
                self.history["proxy_loss"].append(float(graph_info["proxy_loss"]))
                self.history["bc_loss"].append(float(bc_loss))
                self.history["policy_loss"].append(float(policy_loss))
                # [debug doc section 1.2] Keep actor, critic, entropy, and
                # gradient norm separate from policy_loss so actor and critic
                # failures remain distinguishable.
                self.history["actor_loss"].append(
                    float(policy_update_info["actor_loss"])
                )
                self.history["critic_loss"].append(
                    float(policy_update_info["critic_loss"])
                )
                self.history["entropy"].append(
                    float(policy_update_info["entropy"])
                )
                self.history["grad_norm_preclip"].append(
                    float(policy_update_info["grad_norm_preclip"])
                )
                self.history["adv_mean"].append(
                    float(policy_update_info["adv_mean"])
                )
                self.history["adv_std"].append(
                    float(policy_update_info["adv_std"])
                )
                self.history["triggered"].append(int(graph_info["triggered"]))
                self.history["trigger_count"].append(int(trigger_count_now))
                self.history["stage"].append(int(stage_now))
                self.history["proxy_buffer_size"].append(
                    int(self.proxy.get_buffer_size())
                )
                self.history["pushed_proxy_samples"].append(
                    int(pushed_proxy_samples)
                )
                self.history["promoted"].append(int(graph_info["promoted"]))
                self.history["demoted"].append(int(graph_info["demoted"]))
                self.history.setdefault("measured_edge_count", []).append(
                    int(snapshot.get("measured_edge_count", getattr(self, "measured_edge_count", 0)))
                )
                self.history.setdefault("candidate_max_degree", []).append(
                    snapshot.get("candidate_max_degree", getattr(self, "candidate_max_degree", None))
                )
                self.history.setdefault("candidate_construction_subquadratic", []).append(
                    bool(snapshot.get("candidate_construction_subquadratic", False))
                )
                self.history.setdefault("candidate_construction_linear_candidate", []).append(
                    bool(snapshot.get("candidate_construction_linear_candidate", False))
                )
                self.history.setdefault("feature_snapshot_subquadratic", []).append(
                    bool(snapshot.get("feature_snapshot_subquadratic", False))
                )
                self.history.setdefault("feature_snapshot_linear_candidate", []).append(
                    bool(snapshot.get("feature_snapshot_linear_candidate", False))
                )
                self.history.setdefault("candidate_epoch", []).append(
                    int(snapshot.get("candidate_epoch", 0))
                )
                self.history.setdefault("candidate_map_hash", []).append(
                    str(snapshot.get("candidate_map_hash", ""))
                )
                self.history.setdefault("candidate_provider", []).append(
                    str(snapshot.get("candidate_provider", "unknown"))
                )
                self.history.setdefault("candidate_refresh_ms", []).append(
                    float(snapshot.get("candidate_refresh_ms", 0.0))
                )
                self.history.setdefault("candidate_provider_work_units", []).append(
                    int(snapshot.get("candidate_provider_work_units", 0))
                )
                self.history.setdefault("candidate_cell_occupancy_mean", []).append(
                    float(snapshot.get("candidate_cell_occupancy_mean", float("nan")))
                )
                self.history.setdefault("candidate_cell_occupancy_max", []).append(
                    int(snapshot.get("candidate_cell_occupancy_max", -1))
                )
                self.history.setdefault("candidate_added_pairs", []).append(
                    int(snapshot.get("candidate_added_pairs", 0))
                )
                self.history.setdefault("candidate_removed_pairs", []).append(
                    int(snapshot.get("candidate_removed_pairs", 0))
                )
                self.history.setdefault("candidate_churn", []).append(
                    float(snapshot.get("candidate_churn", 0.0))
                )
                self.history.setdefault("E_t", []).append(int(snapshot.get("E_t", 0)))
                self.history.setdefault("K_t", []).append(int(snapshot.get("K_t", 0)))
                self.history.setdefault("d_bar", []).append(float(snapshot.get("d_bar", 0.0)))
                self.history.setdefault("k_bar", []).append(float(snapshot.get("k_bar", 0.0)))
                self.history.setdefault("active_core_pair_count", []).append(
                    int(snapshot.get("active_core_pair_count", 0))
                )
                for _state_key in (
                    "shadow_pair_count", "full_state_pair_count",
                    "belief_pair_count", "signature_pair_count",
                    "degree_limited_ego_count",
                ):
                    self.history.setdefault(_state_key, []).append(
                        int(snapshot.get(_state_key, 0))
                    )
                self.history.setdefault("causal_latency_valid_fraction", []).append(
                    float(snapshot.get("causal_latency_valid_fraction", 0.0))
                )
                self.history.setdefault("causal_latency_onset_valid_fraction", []).append(
                    float(snapshot.get("causal_latency_onset_valid_fraction", 0.0))
                )
                self.history.setdefault("causal_latency_cm_mean", []).append(
                    float(snapshot.get("causal_latency_cm_mean", 0.0))
                )

                print(
                    f"[Final-CIGAMF ep {episode_number:04d}] "
                    f"stage={stage_now} "
                    f"reward={snapshot['mean_reward']:.3f} "
                    f"f1={snapshot['mean_f1']:.3f} "
                    f"f1role={snapshot.get('mean_f1_role', 0.0):.3f} "
                    f"core={snapshot['mean_core_size']:.2f} "
                    f"switch={snapshot['mean_core_switches']:.2f} "
                    f"unc={snapshot['mean_uncertainty']:.5f} "
                    f"mean_mu={snapshot['mean_mu']:.5f} "
                    f"max_p={snapshot['max_p']:.5f} "
                    f"proxy_loss={graph_info['proxy_loss']:.4f} "
                    f"bc_loss={bc_loss:.4f} "
                    f"policy_loss={policy_loss:.4f} "
                    f"trigger={graph_info['triggered']} "
                    f"buffer={self.proxy.get_buffer_size()} "
                    f"pushed={pushed_proxy_samples} "
                    f"promoted={graph_info['promoted']} "
                    f"demoted={graph_info['demoted']} "
                    f"proxy_train_res={graph_info.get('proxy_train_residual', 0.0):.3e} "
                    f"proxy_holdout_res={graph_info.get('proxy_holdout_residual', 0.0):.3e} "
                    f"throughput={snapshot['throughput_agent_steps_per_sec']:.1f} "
                    f"throughput_total={snapshot['throughput_total_agent_steps_per_sec']:.1f}"
                )

        return self.history


class NoTwoTimescaleRunner(FinalCIGAMFRunner):
    """Faithful scheduler-only ablation of Final CIG-AMF.

    Every model, estimator, forced-action propensity, signature tracker,
    auxiliary loss, and policy input is inherited unchanged. The only
    intervention is removal of the slow graph cadence and Stage-0 delay:
    structure modules update every episode from the start.
    """

    ablation_contract = "scheduler_only_graph_update_every_episode"

    def __init__(self, env, cfg, device="cpu"):
        super().__init__(env=env, cfg=cfg, device=device)
        self.scheduler.force_learned_stage()

    def _should_update_graph_this_episode(self):
        return True
