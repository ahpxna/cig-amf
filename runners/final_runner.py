import time
import numpy as np
import torch
import torch.nn.functional as F

from models.structural_proxy import (
    LocalCounterfactualProxyEnsemble,
    build_pair_feat,      # [FIX-X1]
    PAIR_FEAT_DIM,
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


class FinalCIGAMFRunner:
    """
    Final CIG-AMF runner.

    Implements the current methodology:

    1. Population-wide multi-ego training:
        - every agent can be an ego agent;
        - every ego i maintains a separate belief for directed pair (i,j).

    2. Ego-centric directed influence graph:
        - belief_modules[ego] stores b_ij = (mu_bar, sigma_bar, p_core).
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
        self.cfg = dict(cfg)
        self.device = device

        self.n_agents = int(env.n_agents)
        self.obs_dim = int(env.get_obs_dim())
        self.action_dim = int(env.get_action_dim())

        self.core_dim = int(cfg["core_dim"])
        self.periph_dim = int(cfg["periph_dim"])
        self.belief_dim = int(cfg["belief_dim"])

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
            n_horizons=cfg.get("proxy_n_horizons", 3),
            effect_mode=cfg.get("proxy_effect_mode", "signed_aristocrat"),
            use_doubly_robust=cfg.get("proxy_use_doubly_robust", True),
            iw_clip=cfg.get("proxy_iw_clip", 10.0),
            bootstrap_ratio=cfg.get("proxy_bootstrap_ratio", 0.8),
            use_belief_input=cfg.get("proxy_use_belief_input", False),
            ensemble_dropout=cfg.get("proxy_ensemble_dropout", 0.0),
            seed=cfg.get("seed", 0),
            # [FIX-X1] x_ij from Eq. 8. Set zero for the legacy no-x_ij
            # ablation, the configuration that failed H1 over eight seeds.
            pair_feat_dim=cfg.get("proxy_pair_feat_dim", PAIR_FEAT_DIM),
        )

        self.pair_rel_module = PairRelationalModule(
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.core_dim,
            shadow_dim=cfg["shadow_dim"],
            rel_feat_dim=6,
            lr=cfg["core_lr"],
            bc_buffer_size=cfg.get("bc_buffer_size", 200000),
            grad_clip=cfg.get("bc_grad_clip", 1.0),
            shadow_loss_weight=cfg.get("shadow_loss_weight", 0.35),
            device=device,
        )

        self.periph_module = PeripheralMultiMemory(
            action_dim=self.action_dim,
            num_slots=cfg["num_memory_slots"],
            memory_dim=cfg["periph_memory_dim"],
            out_dim=self.periph_dim,
            mu_floor=cfg.get("periph_mu_floor", 0.02),
            beta_floor=cfg.get("periph_beta_floor", 0.05),
            use_uniform_mix=cfg.get("periph_use_uniform_mix", True),
            uniform_mix=cfg.get("periph_uniform_mix", 0.25),
            lb_coeff=cfg.get("periph_lb_coeff", 0.5),
            # [FIX-2] orth_coeff previously ignored cfg and remained at 1e-2.
            # The No-AuxLoss ablation then disabled only L_lb while L_orth
            # remained active, failing to isolate Eq. 26-27 L_aux.
            orth_coeff=cfg.get("periph_orth_coeff", 1e-2),
        ).to(device)

        self.belief_summary_builder = BeliefSummaryBuilder(
            top_k=cfg["belief_top_k"],
            pooled_hidden=cfg["belief_pooled_hidden"],
            out_dim=self.belief_dim,
            priority_mu_floor=cfg.get("belief_priority_mu_floor", 0.02),
        ).to(device)

        self.policy_value = PolicyValueNet(
            obs_dim=self.obs_dim,
            core_dim=self.core_dim,
            peripheral_dim=self.periph_dim,
            belief_dim=self.belief_dim,
            action_dim=self.action_dim,
            hidden=cfg["policy_hidden"],
        ).to(device)

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
            z_threshold=cfg.get("z_threshold", 3.0),
            require_both=cfg.get("require_both", False),
            refractory=cfg.get("refractory", 10),
            inflation_factor=cfg.get("inflation_factor", 2.5),
            inflation_t_reset=cfg.get("inflation_t_reset", 1),
        )
        self.sig_tracker = InfluenceSignatureTracker(
            n_agents=self.n_agents,
            window=cfg.get("sig_tracker_window", 30),
        )
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
        self.heads = EgoConditionedHeads(
            latent_dim=self.pair_rel_module.hidden_dim,
            n_horizons=cfg.get("proxy_n_horizons", 3),
        ).to(self.device)
        self.heads_optim = torch.optim.Adam(
            self.heads.parameters(), lr=cfg.get("heads_lr", cfg.get("core_lr", 5e-4))
        )
        self.drift = DriftDetector(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_horizons=cfg.get("drift_n_horizons", 3),
            warmup_batches=cfg.get("drift_warmup_batches", 200),
            recalibrate_after=cfg.get("drift_recalibrate_after", 15),
            seed=cfg.get("seed", 0),
            device=self.device,
        )
        self.matdet = MatrixDriftDetector(window=cfg.get("matdet_window", 20))
        self.recip = ReciprocityTracker(
            n_agents=self.n_agents,
            min_causal_samples=cfg.get("recip_min_causal_samples", 20),
        )

        self.replay_builder = MultiEgoReplayBuilder(
            discount=cfg["discount"],
            horizon=cfg["causal_horizon"],
        )

        self.belief_modules = {
            ego: BayesLightBeliefState(
                ego_id=ego,
                neighbor_ids=[j for j in range(self.n_agents) if j != ego],
                lambda_0=cfg["belief_lambda_0"],
                uncertainty_scale=cfg["belief_uncertainty_scale"],
                tau=cfg["belief_tau"],
                tau_in=cfg["belief_tau_in"],
                tau_out=cfg["belief_tau_out"],
                weak_prior_top_k=cfg["seed_core_top_k"],
                min_core_size=cfg.get("min_core_size", 1),
                max_core_size=cfg.get("max_core_size", 4),
                sigma_floor=cfg.get("sigma_floor", 0.05),
                # v2 defaults match belief_layer.py recommendations and
                # preserve behaviour when cfg omits them.
                core_rule=cfg.get("belief_core_rule", "lcb"),
                kappa=cfg.get("belief_kappa", 1.0),
                alpha_decay=cfg.get("belief_alpha_decay", 0.7),
                adaptive_k=cfg.get("belief_adaptive_k", False),
                adaptive_k_min=cfg.get("belief_adaptive_k_min", 1),
                signed_balance=cfg.get("belief_signed_balance", 0.5),
            )
            for ego in range(self.n_agents)
        }

        self._initialize_seeded_cores()

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
        }

    # ============================================================
    # Construction helpers
    # ============================================================

    def _make_belief_state(self, ego):
        """
        Create BayesLightBeliefState.

        Includes a fallback for belief_layer.py versions that do not yet accept
        min_core_size or sigma_floor. Updated versions take the first branch.
        """
        common_kwargs = dict(
            ego_id=ego,
            neighbor_ids=[j for j in range(self.n_agents) if j != ego],
            lambda_0=self.cfg["belief_lambda_0"],
            uncertainty_scale=self.cfg["belief_uncertainty_scale"],
            tau=self.cfg["belief_tau"],
            tau_in=self.cfg["belief_tau_in"],
            tau_out=self.cfg["belief_tau_out"],
            weak_prior_top_k=self.cfg["seed_core_top_k"],
        )

        try:
            return BayesLightBeliefState(
                **common_kwargs,
                min_core_size=self.cfg.get("min_core_size", 1),
                sigma_floor=self.cfg.get("sigma_floor", 0.0),
                max_core_size=self.cfg.get("max_core_size",4)
            )
        except TypeError:
            return BayesLightBeliefState(**common_kwargs)

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
        role_bonus = {
            "hauler": 0.20,
            "processor": 0.26,
            "dispatcher": 0.24,
            "sweeper": 0.10,
            "spoiler": 0.08,
        }

        pi = self.env.positions[int(ego)]
        zi = self.env.agent_zone[int(ego)]
        scores = {}

        for j in range(self.n_agents):
            if j == ego:
                continue

            pj = self.env.positions[int(j)]
            zj = self.env.agent_zone[int(j)]
            dist = abs(pj[0] - pi[0]) + abs(pj[1] - pi[1])

            proximity = 1.0 / (1.0 + float(dist))
            same_zone = 0.25 if int(zi) == int(zj) else 0.0
            role_term = role_bonus.get(self.env.agent_role[int(j)], 0.0)

            scores[j] = float(proximity + same_zone + role_term)

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

    def _build_belief_items_for_ego(self, ego):
        belief_state = self.belief_modules[ego].get_state_dict()
        items = []

        for j in range(self.n_agents):
            if j == ego:
                continue

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
            return np.zeros((0, 9), dtype=np.float32)

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

    def _build_periph_inputs_for_ego(self, ego):
        belief_mod = self.belief_modules[ego]
        periph_ids = sorted(list(belief_mod.get_peripheral_set()))
        belief_state = belief_mod.get_state_dict()

        return self.periph_module.build_inputs(
            ego_id=ego,
            peripheral_ids=periph_ids,
            env=self.env,
            belief_state=belief_state,
            prev_core_set=belief_mod.prev_core_set,
        )

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
        )
        return self._periph_summary_np_from_inputs(inputs)

    # ============================================================
    # Action selection
    # ============================================================

    def _select_actions_population(self, obs_all):
        """
        Batched policy forward for the full population.

        Semantics remain unchanged: every ego has distinct B_i, Z_i, and M_i,
        while excluding-j context remains cached by ego/j so proxy samples
        match action-time context.
        """
        actions = {}

        cache = {
            "belief_items_cache": {},
            "belief_summary_cache": {},
            "core_summary_cache": {},
            "periph_inputs_cache": {},
            "periph_summary_cache": {},
            "core_context_excluding": {},
            "periph_context_excluding": {},
            "value_cache": {},
            # [FIX-X1] Snapshot geometry at this timestep so replay_builder can
            # construct x_ij at sample creation time. It cannot be reconstructed
            # after the episode because env.positions then contains final state.
            # Cost is negligible: 24 coordinates and 24 zone IDs per timestep.
            "geom_snapshot": {
                "positions": [list(self.env.positions[i]) for i in range(self.env.n_agents)],
                "agent_zone": [int(z) for z in self.env.agent_zone],
                "grid_size": int(getattr(self.env, "grid_size", 1)),
                "n_zones": int(getattr(self.env, "n_zones", 1)),
            },
        }

        obs_batch = []
        core_batch = []
        periph_batch = []
        belief_batch = []

        for ego in range(self.n_agents):
            belief_items = self._build_belief_items_for_ego(ego)
            periph_inputs = self._build_periph_inputs_for_ego(ego)

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

            for j in range(self.n_agents):
                if j == ego:
                    continue

                cache["core_context_excluding"][ego][j] = self._core_context_excluding(
                    ego,
                    j,
                )

                cache["periph_context_excluding"][ego][j] = self._periph_context_excluding(
                    ego,
                    j,
                )

            obs_batch.append(self.env.get_obs_of_ego(obs_all, ego))
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

        with torch.no_grad():
            logits, values = self.policy_value(
                obs_t,
                core_t,
                periph_t,
                belief_t,
            )

            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            sampled = dist.sample().detach().cpu().numpy()
            # One CPU/GPU synchronization for all values replaces the old
            # n_agents per-step .item() synchronizations.
            values_np = values.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()

        # [EpsilonForcedActionController] This must run after action sampling
        # and before env.step(). apply() mutates forced entries in place and
        # returns effective propensity eps*uniform+(1-eps)*pi, not raw policy
        # probabilities. DR in replay_builder/proxy requires that behaviour
        # probability. Using raw probabilities silently introduces systematic
        # DR bias without an exception or warning.
        actions_list = [int(sampled[ego]) for ego in range(self.n_agents)]
        _pre_forcing = list(actions_list)          # [VERIFY-F1]
        forced_mask, effective_probs = self.forcer.apply(
            actions=actions_list,
            policy_probs=probs_np,
        )

        # ------------------------------------------------------------------
        # [VERIFY-F1] Verify that forced actions survive into the trajectory.
        #
        # Observed min_head_frac was 0.001-0.005, but uniform forcing over six
        # actions gives a theoretical lower bound forced_frac/|A| of
        # 0.013-0.033. A 15-30x deficit is impossible in a correct pipeline and
        # implies that the buffer stores pre-override a_j or applies override
        # after append.
        #
        # Code order appears correct because apply mutates before packing, but
        # inspection alone cannot establish this runtime property. Measure
        # n_forced_seen, n_actually_changed relative to pre-forcing actions,
        # and hist_forced across forced agents. Correct forcing yields an
        # approximately uniform histogram. Positive n_forced_seen with zero
        # actual changes signals a defect.
        # ------------------------------------------------------------------
        try:
            fidx = [k for k in range(self.n_agents) if bool(forced_mask[k])]
            if fidx:
                self._vf1_n_forced = getattr(self, "_vf1_n_forced", 0) + len(fidx)
                self._vf1_n_changed = getattr(self, "_vf1_n_changed", 0) + sum(
                    1 for k in fidx if actions_list[k] != _pre_forcing[k]
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

        cache["forced_mask"] = forced_mask
        cache["behaviour_probs"] = effective_probs
        # Save raw policy probabilities (π_j(a|s)) as well; useful for
        # downstream diagnostics and for runners that expect policy_probs
        # in the trajectory. Shape: [n_agents, A]
        cache["policy_probs"] = probs_np

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
        obs_all = self.env.reset()

        if self.scheduler.in_warmup():
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
        prev_forced_mask = None

        while not done:
            actions_dict, cache = self._select_actions_population(obs_all)
            actions_list = [actions_dict[a] for a in range(self.n_agents)]

            env_snapshot_before_step = self.env.clone_state()
            h_snapshot_before_latent_update = self.pair_rel_module.clone_full_states_np()

            next_obs_all, rewards, done, info = self.env.step(actions_list)
            env_snapshot_after_step = self.env.clone_state()

            rewards = np.array(rewards, dtype=np.float32)
            ep_reward += rewards

            trajectory.append(
                {
                    "obs_all": [x.copy() for x in obs_all],
                    "actions": list(actions_list),
                    "rewards": list(rewards),
                    "belief_items_cache": cache["belief_items_cache"],
                    "behaviour_probs": cache.get("behaviour_probs"),
                    "policy_probs": cache.get("policy_probs"),
                    "belief_summary_cache": cache["belief_summary_cache"],
                    "core_summary_cache": cache["core_summary_cache"],
                    "periph_inputs_cache": cache["periph_inputs_cache"],
                    "periph_summary_cache": cache["periph_summary_cache"],
                    "core_context_excluding": cache["core_context_excluding"],
                    "periph_context_excluding": cache["periph_context_excluding"],
                    "value_cache": cache["value_cache"],
                    "geom_snapshot": cache["geom_snapshot"],   # [FIX-X1]
                    "forced_mask": cache["forced_mask"],
                    
                    "env_snapshot_before_step": env_snapshot_before_step,
                    "env_snapshot_after_step": env_snapshot_after_step,
                    "h_snapshot_before_latent_update": h_snapshot_before_latent_update,
                    "info": info,
                }
            )

            if (
                prev_obs_all is not None
                and prev_actions is not None
                and prev_env_snapshot_before_step is not None
            ):
                self.env.restore_state(prev_env_snapshot_before_step)

                self.pair_rel_module.add_bc_transition(
                    observations={a: prev_obs_all[a] for a in range(self.n_agents)},
                    actions={a: prev_actions[a] for a in range(self.n_agents)},
                    next_actions={a: actions_list[a] for a in range(self.n_agents)},
                    env=self.env,
                    h_prev_snapshot=prev_h_snapshot,
                )

                self.env.restore_state(env_snapshot_after_step)

                # [ReciprocityTracker] Diagnostic only; never feed this into
                # action selection. Use z_ij/s_ij at t, the same context as
                # add_bc_transition, true a_j at t+1, and whether a_i^t was
                # forced. Forcing makes a_i^t mechanically independent, so any
                # predictive gain for a_j^{t+1} is causal rather than shared-
                # observation confounding.
                if prev_forced_mask is not None:
                    self._record_reciprocity(
                        prev_h_snapshot=prev_h_snapshot,
                        actions_list=actions_list,
                        prev_forced_mask=prev_forced_mask,
                    )

            self.env.restore_state(env_snapshot_before_step)

            self.pair_rel_module.step_population(
                obs_all=obs_all,
                actions=actions_list,
                env=self.env,
            )

            self.env.restore_state(env_snapshot_after_step)

            prev_obs_all = [x.copy() for x in obs_all]
            prev_actions = list(actions_list)
            prev_env_snapshot_before_step = env_snapshot_before_step
            prev_h_snapshot = h_snapshot_before_latent_update
            prev_forced_mask = cache["forced_mask"]

            obs_all = next_obs_all

        runtime = time.time() - t0

        # Advance the epsilon annealing schedule exactly once per episode; see
        # step_episode(). VERIFY-F1 reports a small conclusive per-episode set.
        if getattr(self, "_vf1_n_forced", 0) > 0:
            h = self._vf1_hist.astype(np.float64)
            h = h / max(h.sum(), 1.0)
            # The forcer is targeted: per-agent epsilon differs and smoke test
            # section 5 measured 6.1x concentration, so forced-agent identity
            # is nonuniform. Conditional on forcing, the chosen action must be
            # uniform over |A|; this determines the min_head_frac lower bound.
            print(
                f"[VERIFY-F1] forced_seen={self._vf1_n_forced} "
                f"actually_changed={self._vf1_n_changed} "
                f"({100.0*self._vf1_n_changed/max(1,self._vf1_n_forced):.1f}%; "
                f"kỳ vọng ~{100.0*(1-1.0/self.action_dim):.0f}% vì ép trùng "
                f"action cũ với xác suất 1/|A|) "
                f"hist_action_forced={np.round(h, 3).tolist()}"
            )
            self._vf1_n_forced = 0
            self._vf1_n_changed = 0
            self._vf1_hist = np.zeros(int(self.action_dim), dtype=np.int64)

        self.forcer.step_episode()

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
            for j in range(self.n_agents)
            if j != ego
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

        returns = [[0.0 for _ in range(self.n_agents)] for _ in range(T)]
        R = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(T)):
            R = (
                np.array(trajectory[t]["rewards"], dtype=np.float32)
                + self.cfg["discount"] * R
            )

            for ego in range(self.n_agents):
                returns[t][ego] = float(R[ego])

        total_loss = 0.0
        total_periph_aux_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        adv_mean_acc = []
        adv_std_acc = []
        count = 0

        for t, step in enumerate(trajectory):
            obs_batch = []
            core_batch = []
            actions_batch = []
            returns_batch = []
            belief_tensors = []
            periph_tensors = []

            for ego in range(self.n_agents):
                obs_i = self.env.get_obs_of_ego(step["obs_all"], ego)

                obs_batch.append(obs_i)
                core_batch.append(step["core_summary_cache"][ego])
                actions_batch.append(int(step["actions"][ego]))
                returns_batch.append(float(returns[t][ego]))

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
            )

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
            adv_raw = (ret_t - value).detach()

            adv_mean_acc.append(float(adv_raw.mean().item()))
            adv_std_acc.append(float(adv_raw.std(unbiased=False).item()))

            adv = (adv_raw - adv_raw.mean()) / (adv_raw.std(unbiased=False) + 1e-8)

            policy_loss = -logp * adv
            value_loss = F.mse_loss(value, ret_t, reduction="none")
            entropy = dist.entropy()

            loss_vec = policy_loss + 0.5 * value_loss - 0.01 * entropy

            total_loss = total_loss + loss_vec.sum()
            total_actor_loss = total_actor_loss + policy_loss.sum()
            total_critic_loss = total_critic_loss + value_loss.sum()
            total_entropy = total_entropy + entropy.sum()
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
            0.5,
        )

        self.policy_optim.step()

        return {
            # [FIX-8] The old loss included auxiliary terms, so aux dominated
            # policy_loss: about 4.4 of 4.58. Keep it separate to satisfy the
            # debug rule against diagnosing aggregated values.
            "loss": float(policy_term.item()),
            "aux_loss": float(aux_term.item()) if torch.is_tensor(aux_term) else float(aux_term),
            "total_loss_with_aux": float(loss.item()),
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
        pushed = self.replay_builder.push_trajectory_to_proxy(
            trajectory=trajectory,
            proxy_ensemble=self.proxy,
            env=self.env,
        )

        return int(pushed)

    def _score_all_pairs_and_update_beliefs(self, obs_all, actions, observed_returns=None, behaviour_probs=None):
        """
        Score every directed pair (ego,j) and update Bayes-light beliefs.

        Semantics remain unchanged: use proxy-ensemble scores with context
        Z_i^{-j}, M_i^{-j}, B_i; update beliefs followed by hysteresis; and
        warm-start z_ij from shadow when j enters the core.
        """
        total_promoted = 0
        total_demoted = 0

        # Signed [n_agents,n_agents] influence matrix W[ego,j]=mu_ij for the
        # MatrixDriftDetector, an independent second trigger.
        influence_matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)

        for ego in range(self.n_agents):
            obs_i = self.env.get_obs_of_ego(obs_all, ego)
            action_i = int(actions[ego])

            belief_items = self._build_belief_items_for_ego(ego)
            belief_summary = self._belief_summary_np_from_items(belief_items)

            obs_i_batch = []
            action_i_batch = []
            action_j_batch = []
            z_batch = []
            m_batch = []
            b_batch = []
            neighbor_ids = []

            for j in range(self.n_agents):
                if j == ego:
                    continue

                obs_i_batch.append(obs_i)
                action_i_batch.append(action_i)
                action_j_batch.append(int(actions[j]))
                z_batch.append(self._core_context_excluding(ego, j))
                m_batch.append(self._periph_context_excluding(ego, j))
                b_batch.append(belief_summary)
                neighbor_ids.append(j)

            # Prepare optional DR inputs: observed_returns and behaviour_probs
            observed_returns_batch = None
            behaviour_probs_obs_batch = None

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

            # score_batch_full accepts optional DR inputs without changing
            # output when they are absent.
            out = self.proxy.score_batch_full(
                obs_i_batch=obs_i_batch,
                action_i_batch=action_i_batch,
                observed_action_j_batch=action_j_batch,
                z_core_excl_j_batch=z_batch,
                m_periph_excl_j_batch=m_batch,
                belief_summary_batch=b_batch,
                policy_probs_j_batch=None,
                observed_returns_batch=observed_returns_batch,
                behaviour_probs_obs_batch=behaviour_probs_obs_batch,
                # [FIX-X1] Build x_ij from current online environment state
                # using the same function as the push path to prevent
                # train/serve skew.
                pair_feat_batch=[
                    build_pair_feat(
                        self.env.positions, self.env.agent_zone,
                        getattr(self.env, "grid_size", 1),
                        getattr(self.env, "n_zones", 1), ego, j,
                    )
                    for j in neighbor_ids
                ],
            )
            mu_arr, sigma_arr = out["mu"], out["sigma"]

            # [InfluenceSignatureTracker] Previously instantiated without any
            # update call, leaving signature/get_role empty. context_key is j's
            # current zone and supports context_std for conditional influence.
            self.sig_tracker.update_from_proxy_output(
                ego_id=ego,
                neighbor_ids=neighbor_ids,
                proxy_out=out,
                context_keys=[int(self.env.agent_zone[j]) for j in neighbor_ids],
            )

            mu_sigma = {
                j: (float(mu_arr[k]), float(sigma_arr[k]))
                for k, j in enumerate(neighbor_ids)
            }

            for k, j in enumerate(neighbor_ids):
                influence_matrix[ego, j] = float(mu_arr[k])

            update_result = self.belief_modules[ego].update_batch(mu_sigma)

            if update_result is None:
                promoted = set()
                demoted = set()
            else:
                promoted, demoted = update_result

            self.pair_rel_module.warm_start_if_promoted(ego, promoted)

            total_promoted += len(promoted)
            total_demoted += len(demoted)

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
            h_returns = self.replay_builder.build_h_step_returns(trajectory, self.n_agents)
        except Exception:
            h_returns = [None for _ in range(len(trajectory))]

        promoted = 0
        demoted = 0
        last_influence_matrix = None

        for idx in step_indices:
            step = trajectory[int(idx)]
            self.env.restore_state(step["env_snapshot_before_step"])
            observed_returns = h_returns[int(idx)] if h_returns is not None else None
            behaviour_probs = step.get("behaviour_probs") if isinstance(step, dict) else None

            p_i, d_i, w_i = self._score_all_pairs_and_update_beliefs(
                obs_all=step["obs_all"],
                actions=step["actions"],
                observed_returns=observed_returns,
                behaviour_probs=behaviour_probs,
            )
            promoted += int(p_i)
            demoted += int(d_i)
            last_influence_matrix = w_i

        self.env.restore_state(trajectory[-1]["env_snapshot_after_step"])

        residual = self.proxy.get_latest_residual()

        # Two independent evaluate_drift triggers replace v1's self-
        # contaminated record_structural_residual signal.

        # Trigger 1: frozen blinded probe reading proxy.buffer directly.
        drift_info = self.drift.step(
            episode=int(self.scheduler.episode),
            buffer=self.proxy.buffer,
            n_train_batches=self.cfg.get("drift_train_batches", 5),
        )
        probe_z = float(drift_info.get("z", 0.0) or 0.0)

        # Trigger 2: jump in the signed influence matrix.
        matrix_z = 0.0
        if last_influence_matrix is not None:
            self.matdet.update(last_influence_matrix)
            matrix_z = float(self.matdet.z_score())

        fire_info = self.scheduler.evaluate_drift(
            probe_z=probe_z,
            matrix_z=matrix_z,
            belief_modules=self.belief_modules,
            drift_detector=self.drift,
        )
        if fire_info["fired"]:
            print(f"[DRIFT-FIRE] ep={self.scheduler.episode} reason={fire_info['reason']} n_inflated={fire_info['n_inflated']} "
                  f"probe_z={fire_info['probe_z']:.2f} matrix_z={fire_info['matrix_z']:.2f}")
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
        mu_vals = []
        p_vals = []

        for ego in range(self.n_agents):
            state = self.belief_modules[ego].get_state_dict()

            for _, item in state.items():
                mu_vals.append(float(item["mu_bar"]))
                p_vals.append(float(item["p_core"]))

        mean_mu = float(np.mean(np.abs(mu_vals))) if len(mu_vals) > 0 else 0.0
        max_p = float(np.max(p_vals)) if len(p_vals) > 0 else 0.0

        return mean_mu, max_p

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
        # [F1-TOPK] Core-F1 ground truth is measured top-k |Phi|, not static
        # role labels.
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
                    k = max(1, len(pred_core)) if pred_core else self.cfg.get(
                        "seed_core_top_k", 3)
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

        mean_mu, max_p = self._belief_population_stats()

        return {
            "mean_reward": float(np.mean(episode_reward)),
            "reward_per_agent": float(np.mean(episode_reward)),
            "mean_f1": float(np.mean(f1s)),
            "mean_f1_role": float(np.mean(f1s_role)) if f1s_role else 0.0,
            "mean_temporal_var": float(np.mean(tvars)),
            "mean_uncertainty": float(np.mean(uncs)),
            "mean_core_size": float(np.mean(core_sizes)),
            "mean_core_switches": float(np.mean(core_switches)),
            "mean_mu": float(mean_mu),
            "max_p": float(max_p),
            "runtime": float(rollout_runtime),
            "throughput_agent_steps_per_sec": float(throughput),
            "episode_runtime_total": float(total_runtime),
            "throughput_total_agent_steps_per_sec": float(throughput_total),
        }

    # ============================================================
    # Main loop
    # ============================================================

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
        for ep in range(int(n_episodes)):
            stage_before_episode_step = int(self.scheduler.stage)

            trajectory, episode_reward, runtime = self.collect_episode()

            pushed_proxy_samples = self.push_trajectory_to_proxy_buffer(trajectory)

            t_policy = time.time()
            policy_update_info = self.update_policy(trajectory)
            policy_loss = policy_update_info["loss"]
            policy_runtime = time.time() - t_policy

            t_bc = time.time()
            bc_loss = self.pair_rel_module.train_bc(
                n_steps=self.cfg["bc_train_steps"],
                batch_size=self.cfg["bc_batch_size"],
                # [ego_conditioned_latent.py] Add E1/E2 to the same train_bc
                # loss/backward so z_ij is forced to carry ego information.
                # docstring train_bc trong core_behavior.py.
                heads=self.heads,
                heads_optim=self.heads_optim,
                w_contrastive=self.cfg.get("heads_w_contrastive", 0.3),
                w_influence=self.cfg.get("heads_w_influence", 1.0),
                w_target_fn=lambda ego_id, nb_id: self.belief_modules[
                    ego_id
                ].debiased_mu(nb_id),
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

            if self.scheduler.should_update_graph():
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

# Runner calibration patch.
            import traceback

            # 1. Initialize counters and the persistent multiplier.
            self._calib_fail_count = getattr(self, '_calib_fail_count', 0)
            self._consecutive_zero_max_p = getattr(self, '_consecutive_zero_max_p', 0)
            self._kappa_multiplier = getattr(self, '_kappa_multiplier', 1.0) 

            # 2. CẬP NHẬT CỜ REACTIVE
            try:
                current_max_p = max([float(np.max(b_mod._p_core_arr)) for b_mod in self.belief_modules.values() if hasattr(b_mod, '_p_core_arr')])
            except:
                current_max_p = 1.0

            # Require 0.05: below a 5% entry probability indicates core collapse.
            if current_max_p <= 0.05:
                self._consecutive_zero_max_p += 1
            else:
                self._consecutive_zero_max_p = 0

            # 3. Define trigger conditions.
            is_stage_transition = (stage_before_episode_step == 0 and stage_after_episode_step == 1)
            is_periodic_calib = (stage_after_episode_step == 1 and ep > 0 and (ep % 5 == 0 if ep <= 30 else ep % 25 == 0))
            is_reactive_calib = (self._consecutive_zero_max_p >= 3)
            
            # Update the multiplier while retaining collapse state.
            if is_reactive_calib:
                # During collapse, halve kappa immediately with floor 0.01.
                self._kappa_multiplier = max(0.01, self._kappa_multiplier * 0.5) 
            elif is_periodic_calib and current_max_p > 0.1:
                # During healthy periodic runs (>10%), recover kappa gradually.
                self._kappa_multiplier = min(1.0, self._kappa_multiplier * 1.5)

            if is_stage_transition:
                self._reset_switch_counters_if_available()

            # 4. Perform calibration.
            if is_stage_transition or is_periodic_calib or is_reactive_calib:
                calib_reason = "Stage Transition" if is_stage_transition else ("Periodic" if is_periodic_calib else "Reactive (max_p=0)")
                print(f"\n[STEP 0 DIAGNOSTIC] --- Threshold Calibration Triggered at Ep {ep} | Reason: {calib_reason} ---")

                try:
                    calib = self.sig_tracker.auto_calibrate()
                    tau_role = calib["tau_role"]

                    all_sigma_bar = [
                        float(v.get("sigma_bar", 0.0))
                        for ego in range(self.n_agents)
                        for v in self.belief_modules[ego].get_state_dict().values()
                    ]
                    if len(all_sigma_bar) >= 4:
                        arr = np.asarray(all_sigma_bar)
                        sigma_hi = float(np.percentile(arr, 80.0))
                        sigma_p50 = float(np.percentile(arr, 50.0))
                        sigma_p90 = float(np.percentile(arr, 90.0))
                        sigma_iqr = float(np.percentile(arr, 75.0) - np.percentile(arr, 25.0))
                    else:
                        sigma_hi = calib["sigma_hi"]  # Fallback before enough real-belief samples.
                        sigma_p50 = sigma_p90 = sigma_iqr = float("nan")
                    g_anom_mean = float(getattr(self.periph_module, "g_anom_usage_ema", torch.zeros(1)).item())
                    print(f"   [SLOT-DEBUG] sigma_p50={sigma_p50:.6f} sigma_p90={sigma_p90:.6f} "
                          f"sigma_hi={sigma_hi:.6f} g_anom_mean={g_anom_mean:.4f}")
                    print(f"   [VERIFY] sigma_hi={sigma_hi:.6f}  tau_role={tau_role:.6f}")

                    if hasattr(self, 'periph_module'):
                        # [B2.3] Pass sigma_iqr too. It was previously omitted,
                        # causing sigma_iqr_floor to equal sigma_hi and k_sg to explode.
                        self.periph_module.set_role_thresholds(
                            tau_role, sigma_hi,
                            sigma_iqr=(
                                sigma_iqr if np.isfinite(sigma_iqr) else None
                            ),
                        )

                    modules_list = list(self.belief_modules.values()) if isinstance(self.belief_modules, dict) else list(self.belief_modules)

                    _tau, _sig, _kap = 0.0, 0.0, 0.0
                    for b_mod in modules_list:
                        if not hasattr(b_mod, 'tau'):
                            continue
                        
                        b_mod.tau = max(1e-4, tau_role * 0.5)
                        b_mod.sigma_floor = max(1e-4, sigma_hi * 0.1)
                        
                        # Compute base kappa and apply the persistent multiplier.
                        base_kappa = float(np.clip(tau_role / (sigma_hi + 1e-8), 0.05, 2.0))
                        b_mod.kappa = base_kappa * self._kappa_multiplier
                        
                        _tau, _sig, _kap = b_mod.tau, b_mod.sigma_floor, b_mod.kappa

                    print(f"   [SYNC] Param Update: tau={_tau:.5f}, sigma_floor={_sig:.5f}, kappa={_kap:.5f} (Multiplier: {self._kappa_multiplier:.3f})")

                    target_mod = modules_list[0]
                    dbg = getattr(target_mod, '_last_lcb_debug', None)
                    if dbg:
                        print(f"   [MATH CHECK] |mu_deb| mean={dbg.get('mu_deb_mean', float('nan')):.6f}"
                              f" | Penalty mean={dbg.get('penalty_mean', float('nan')):.6f}"
                              f" | p mean={dbg.get('p_mean', float('nan')):.4f}")

                    try:
                        sample_p = np.atleast_1d(target_mod._p_core_arr).flatten()[:5]
                        print(f"   [SANITY] p_core_arr[0:5] hiện tại: {sample_p}")
                    except Exception as sanity_e:
                        pass

                    self._calib_fail_count = 0
                    if is_reactive_calib:
                        self._consecutive_zero_max_p = 0

                except Exception as e:
                    self._calib_fail_count += 1
                    print(f"   [ERROR] Calibration failed (Count: {self._calib_fail_count}/3): {e}")
            # End runner calibration patch.

            if ep % int(eval_every) == 0:
                self.history["episodes"].append(ep)
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
                # v2 removed internal EWMA/CUSUM in favor of evaluate_drift
                # with probe_z/matrix_z. Preserve history columns using the
                # latest z-scores for storage-format compatibility.
                self.history["scheduler_residual_ewma"].append(
                    float(graph_info.get("probe_z", 0.0) or 0.0)
                )
                self.history["scheduler_cusum_score"].append(
                    float(graph_info.get("matrix_z", 0.0) or 0.0)
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

                print(
                    f"[Final-CIGAMF ep {ep:04d}] "
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
