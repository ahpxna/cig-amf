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

    Bám đúng methodology hiện tại:

    1. Population-wide multi-ego training:
        - mọi agent đều có thể là ego-agent.
        - mỗi ego-agent i giữ belief riêng trên các directed pair (i, j).

    2. Ego-centric directed influence graph:
        - belief_modules[ego] lưu b_ij = (mu_bar, sigma_bar, p_core).
        - core/peripheral partition là riêng theo ego.

    3. Pair-specific relational latent:
        - z_ij không phải z_j global.
        - z_ij encode relation giữa behaviour của j và ego i.
        - update z_ij phải dùng context tại thời điểm action selection:
              o_i^t, o_j^t, a_i^t, a_j^t, Delta_ij^t.
        - Vì vậy collect_episode() restore env về snapshot trước step
          trước khi gọi pair_rel_module.step_population().

    4. Shadow warm-start:
        - mọi pair có shadow state s_ij.
        - khi j promoted vào core của i, z_ij được warm-start từ s_ij.

    5. Local counterfactual proxy:
        - supervised target là finite-horizon return R_i^(H).
        - proxy sample dùng context excluding-j cache tại thời điểm policy chọn action:
              s_i, a_i, a_j, Z_i^{-j}, M_i^{-j}, B_i.
        - replay/proxy buffer được collect mọi episode, kể cả Stage 0.

    6. Bayes-light belief:
        - proxy ensemble score directed pairs.
        - belief update dùng mean influence và uncertainty scale.
        - core update dùng hysteresis.

    7. Two-stage two-timescale:
        - Stage 0: seeded-core warm-up, collect replay, train policy/value/pair latent.
        - Stage 1: learned belief takeover, proxy/belief/core update định kỳ.
        - residual EWMA/CUSUM trigger tăng tạm thời tần suất structural update.

    8. Evaluation:
        - mean reward.
        - structural F1 nếu env cung cấp diagnostic_core_by_ego.
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
            # ---- v2, default = giá trị gốc trong structural_proxy.py ->
            # KHÔNG đổi hành vi nếu cfg không set ----
            n_horizons=cfg.get("proxy_n_horizons", 3),
            effect_mode=cfg.get("proxy_effect_mode", "signed_aristocrat"),
            use_doubly_robust=cfg.get("proxy_use_doubly_robust", True),
            iw_clip=cfg.get("proxy_iw_clip", 10.0),
            bootstrap_ratio=cfg.get("proxy_bootstrap_ratio", 0.8),
            use_belief_input=cfg.get("proxy_use_belief_input", False),
            ensemble_dropout=cfg.get("proxy_ensemble_dropout", 0.0),
            seed=cfg.get("seed", 0),
            # [FIX-X1] x_ij (Eq 8). Đặt 0 để quay lại hành vi cũ (ablation
            # "no-x_ij" — chính là cấu hình đã làm H1 thất bại ở 8 seed).
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
            # [FIX-2] orth_coeff trước đây KHÔNG đọc từ cfg -> luôn kẹt ở
            # default 1e-2. Ablation "No-AuxLoss" (Eq 26-27) chỉ tắt được L_lb,
            # còn L_orth vẫn chạy => ablation không cô lập đúng L_aux.
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
            # [FIX-P1] default None: cap phá "b_j known exactly" (xem
            # intervention.py) và gần như không bao giờ chạm ở eps=0.03/n=24.
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
                # ---- v2, default = giá trị "khuyến nghị" gốc trong
                # belief_layer.py -> KHÔNG đổi hành vi nếu cfg không set ----
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
            "mean_f1_role": [],   # [F1-TOPK] chấm theo nhãn role (đối chứng)
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
            # [debug doc mục 1.2 -- schema log mới, "0.2 Không bao giờ debug
            # bằng số trung bình"] policy_loss cũ là actor+critic+entropy
            # gộp làm một -- không tách được cái nào hỏng.
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
        Tạo BayesLightBeliefState.

        Có fallback để không crash nếu belief_layer.py của bản hiện tại chưa nhận
        min_core_size hoặc sigma_floor. Nếu file belief_layer.py đã được thay bằng
        bản mới thì nhánh đầu tiên sẽ chạy.
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
        Weak structural prior, không phải ground truth.

        Stage 0 có thể dùng:
        - proximity.
        - same-zone bonus.
        - role relevance.
        - environment-specific local cues.

        Prior này chỉ dùng để seed core lúc warm-up. Sang Stage 1, learned belief
        takeover sẽ quyết định core/peripheral partition.
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
        Dùng sau khi Stage 0 chuyển sang Stage 1.

        Lý do:
        - Stage 0 có thể re-seed core mỗi episode theo weak prior.
        - Nếu env reset positions, weak prior đổi, core switch count bị nhiễu bởi
          bootstrap chứ không phản ánh learned belief instability.
        - Nếu belief layer có reset_switch_counter(), gọi nó.
        - Nếu không có, bỏ qua để không phá backward compatibility.
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
        """Bản đầy đủ — dùng trong training loop để aux_loss thực sự có gradient."""
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
        [GPU_OPTIMIZATION_CONTRACT.md mục 2.1] Bản cũ gọi get_core_summary()
        (mean pooling từ đầu) cho MỖI (ego, j) -> O(core_size) việc lặp lại
        N lần mỗi ego, N^2 lần mỗi timestep. get_core_summary_excluding_all
        đã tồn tại sẵn trong core_behavior.py (thủ thuật sum-trừ-một, CHỈ
        đúng với mean pooling — đúng loại pooling paper đang dùng) nhưng
        chưa được gọi ở đây. Cache theo (ego, core_set hiện tại): 1 lệnh
        sum-minus-one cho CẢ N neighbour của ego đó, N lệnh tra dict còn
        lại chỉ là O(1) lookup.
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

        # Fallback an toàn (không nên xảy ra: exclude_j luôn là neighbour
        # hợp lệ) — giữ đường chậm cũ để KHÔNG BAO GIỜ trả sai kết quả.
        reduced = [x for x in core_set if x != exclude_j]
        return self.pair_rel_module.get_core_summary(ego, reduced)

    def _periph_context_excluding(self, ego, exclude_j):
        """
        [GPU_OPTIMIZATION_CONTRACT.md mục 2.1] Bản cũ: build_inputs() +
        forward ĐẦY ĐỦ (chạy lại item_encoder/slot_router cho gần hết tập)
        RIÊNG cho mỗi (ego, j) -> N lần mỗi ego. forward_excluding_all()
        tính num/den đầy đủ MỘT LẦN rồi trừ-một vector hoá cho cả N
        neighbour cùng lúc (sum-trừ-một, chỉ đúng weighted-mean pooling —
        đã xác nhận đúng loại paper dùng, xem docstring của hàm đó).
        Cache theo (ego, peripheral_set hiện tại); đồng bộ CPU MỘT LẦN
        cho cả batch thay vì mỗi exclude_j một lần .cpu().numpy().
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

        # Fallback an toàn (không nên xảy ra: exclude_j luôn trong peripheral
        # set hiện tại) — đường chậm cũ, KHÔNG BAO GIỜ trả sai kết quả.
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
        Batch policy forward cho toàn population.

        Không đổi logic:
        - mỗi ego có B_i riêng.
        - mỗi ego có Z_i riêng.
        - mỗi ego có M_i riêng.
        - context excluding-j vẫn cache theo ego/j để proxy sample khớp action-time context.
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
            # [FIX-X1] Ảnh chụp hình học TẠI timestep này, để replay_builder
            # dựng x_ij đúng thời điểm mẫu được sinh ra. Không thể tính lại
            # sau episode vì env.positions lúc đó đã là state cuối.
            # Chi phí: 24 toạ độ + 24 zone id / timestep — không đáng kể.
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
            # Một sync CPU<->GPU cho toàn bộ values thay vì gọi .item()
            # từng agent một (bản cũ: n_agents lần đồng bộ mỗi bước).
            values_np = values.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()

        # [EpsilonForcedActionController — intervention.py] PHẢI đứng SAU
        # khi đã sample action, TRƯỚC khi actions được dùng để env.step().
        # apply() sửa actions_list IN-PLACE tại vị trí bị ép, và trả về
        # effective_probs = propensity hiệu dụng (mixture eps*uniform +
        # (1-eps)*pi), KHÔNG PHẢI policy_probs thô — đây là behaviour
        # probability đúng cần cho DR trong replay_builder/proxy. Lấy nhầm
        # policy_probs (bỏ qua forcing) làm propensity sẽ khiến DR chệch có
        # hệ thống một cách im lặng (không crash, không warning).
        actions_list = [int(sampled[ego]) for ego in range(self.n_agents)]
        _pre_forcing = list(actions_list)          # [VERIFY-F1]
        forced_mask, effective_probs = self.forcer.apply(
            actions=actions_list,
            policy_probs=probs_np,
        )

        # ------------------------------------------------------------------
        # [VERIFY-F1] Kiểm chứng forced action SỐNG SÓT tới trajectory.
        #
        # Lý do phải đo: min_head_frac quan sát được là 0.001-0.005, trong khi
        # nếu forced action là uniform trên |A|=6 thì CHẶN DƯỚI LÝ THUYẾT là
        # forced_frac/|A| ~ 0.013-0.033. Thấp hơn chặn dưới 15-30x là BẤT KHẢ
        # THI nếu pipeline đúng => hoặc a_j ghi vào buffer là action TRƯỚC
        # override, hoặc override xảy ra sau khi append.
        #
        # Đọc thứ tự code thì hiện tại có vẻ ĐÚNG (apply mutate in-place rồi
        # actions_list mới được đóng gói). Nhưng "có vẻ đúng" không phải bằng
        # chứng — đây là đúng loại lỗi mà đọc code không bắt được. Đo thật:
        #   n_forced_seen      : số agent forcer báo đã ép
        #   n_actually_changed : số agent action THỰC SỰ đổi so với pre-forcing
        #   hist_forced        : phân bố action TRÊN CÁC AGENT BỊ ÉP
        # Nếu forcing đúng thì hist_forced phải xấp xỉ uniform trên |A|.
        # Nếu n_actually_changed == 0 trong khi n_forced_seen > 0 => bug.
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
        - Therefore we restore env_snapshot_before_step before step_population(),
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

                # [ReciprocityTracker — reciprocity.py] Đúng thiết kế trong
                # file: CÔNG CỤ CHẨN ĐOÁN, không cắm vào action selection.
                # Dùng z_ij/s_ij tại t (cùng ngữ cảnh add_bc_transition vừa
                # dùng) + a_j thật tại t+1 + cờ ego_was_forced tại t (a_i^t
                # bị eps-forcing ép hay không -> a_i^t độc lập cơ học với
                # mọi thứ khác, nên "biết a_i^t giúp đoán a_j^{t+1}" mới là
                # nhân quả thật, không phải confounding từ cùng quan sát).
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

        # [EpsilonForcedActionController] lịch trình anneal eps -> gọi
        # đúng một lần sau mỗi episode (xem docstring step_episode()).
        # [VERIFY-F1] báo cáo mỗi episode — chỉ vài dòng, đủ để kết luận.
        if getattr(self, "_vf1_n_forced", 0) > 0:
            h = self._vf1_hist.astype(np.float64)
            h = h / max(h.sum(), 1.0)
            # LƯU Ý: forcer là TARGETED (eps per-agent khác nhau, smoke test
            # §5 đo được tập trung 6.1x), nên "agent nào bị ép" KHÔNG đều.
            # Nhưng "ép rồi thì chọn action nào" PHẢI đều trên |A| — chính
            # phân bố này quyết định chặn dưới của min_head_frac.
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
        [reciprocity.py] Ghi nhận information gain g_ij (i -> j) cho MỌI
        cặp có hướng, dùng CHÍNH bc_head/shadow_to_full đã có (không train
        thêm gì ở đây, chỉ forward dưới no_grad để chẩn đoán). Batch một
        lần duy nhất cho cả n_agents*(n_agents-1) cặp thay vì gọi bc_head
        riêng từng cặp.
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
        [docs/CIG-AMF_training_debug_master.md mục 2.2(b)/2.2(c)] Hai sửa:

        (b) Advantage KHÔNG chuẩn hoá -- bản cũ dùng thẳng
            `adv = ret_t - value` làm hệ số nhân cho -logp, nên gradient
            scale bám theo reward scale (biến động mạnh trên horizon dài).
            Chuẩn hoá adv về mean 0 / std 1 TRÊN TỪNG BATCH timestep
            (n_agents=24 mẫu mỗi t -- đủ lớn để ước lượng std thô, và
            không cần forward pass thứ hai để chuẩn hoá toàn episode).
            ret_t (target của critic) giữ NGUYÊN thang gốc -- chỉ adv dùng
            cho actor mới bị chuẩn hoá.

        (c)/1.2 policy_loss trước đây là MỘT số gộp actor+critic+entropy
            -> không tách được actor hỏng hay critic hỏng. Giờ tách riêng
            actor_loss/critic_loss/entropy/grad_norm_preclip, trả về dict
            thay vì float trần, để log đúng schema mục 1.2 của debug doc.
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

                # [FIX-CRIT-1] Khối này TRƯỚC ĐÂY nằm NGOÀI vòng lặp ego (dedent
                # một cấp), nên nó dùng biến `ego` rò rỉ = n_agents-1: chỉ tính
                # M_i cho ĐÚNG MỘT ego cuối cùng, rồi PolicyValueNet.forward()
                # âm thầm .expand(B,-1) tensor [1,D] đó ra cho cả 24 agent.
                # Hậu quả (giải thích 3 triệu chứng đang treo):
                #   1. Vi phạm Eq (25): M_i phải ego-specific, thực tế mọi agent
                #      dùng chung peripheral memory của agent 23.
                #   2. slot_usage_ema chỉ được cập nhật từ 1 ego mỗi timestep
                #      => phân phối routing nghèo nàn => usage entropy thấp
                #      (0.44 < 0.5) dù cơ chế semantic slot đã đúng.
                #   3. aux_loss (Eq 26-27) chỉ bằng 1/24 độ lớn dự kiến và chỉ
                #      từ một ego => ablation No-AuxLoss gần như không đổi kết
                #      quả ("byte-identical" với Full-CIGAMF).
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

        # [FIX-8] Sau FIX-CRIT-1, aux_loss được cộng 24 lần/timestep thay vì 1
        # => trọng số HIỆU DỤNG của L_aux tăng đúng 24x so với lúc λ_lb=1.2
        # được tinh chỉnh. Bằng chứng: policy_loss log nhảy 0.19 -> 4.58 (~24x)
        # và reward ep50 tụt xuống -0.711. Thang hiện tại mới là thang ĐÚNG
        # (mean trên mỗi agent-step, khớp với total_loss), nên không sửa mẫu số
        # mà phải hạ λ (xem periph_lb_coeff trong default_cfg).
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
            # [FIX-8] "loss" trước đây GỘP cả aux -> cột policy_loss trong log
            # bị aux nuốt (4.58 trong đó ~4.4 là aux). Vi phạm nguyên tắc 0.2
            # của debug doc ("không debug bằng số gộp"). Tách hẳn ra.
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

        Đây là điểm rất quan trọng:
        - Stage 0 không graph-update.
        - Nhưng Stage 0 phải collect proxy buffer.
        - Nếu chỉ push khi should_update_graph(), proxy sẽ thiếu dữ liệu warm-up.
        """
        pushed = self.replay_builder.push_trajectory_to_proxy(
            trajectory=trajectory,
            proxy_ensemble=self.proxy,
            env=self.env,
        )

        return int(pushed)

    def _score_all_pairs_and_update_beliefs(self, obs_all, actions, observed_returns=None, behaviour_probs=None):
        """
        Score mọi directed pair (ego, j) và update Bayes-light belief.

        Không đổi logic:
        - score bằng proxy ensemble.
        - context là Z_i^{-j}, M_i^{-j}, B_i.
        - belief update rồi hysteresis.
        - nếu j promoted vào core thì warm-start z_ij từ shadow.
        """
        total_promoted = 0
        total_demoted = 0

        # Ma trận ảnh hưởng có dấu [n_agents, n_agents], W[ego, j] = mu_ij.
        # Dùng cho MatrixDriftDetector (cò súng thứ hai, độc lập với probe).
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

            # score_batch_full thêm các input DR nếu có, không đổi output khi absent
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
                # [FIX-X1] x_ij dựng từ state HIỆN TẠI của env (score path
                # chạy online), cùng hàm với push path -> không train/serve skew.
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

            # [InfluenceSignatureTracker — influence_signature.py] Trước đây
            # tạo xong không ai gọi update() -> signature/get_role() luôn
            # rỗng. context_key = zone hiện tại của j, dùng cho chiều
            # context_std (ảnh hưởng có điều kiện theo tình huống hay không).
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

        Không push replay ở đây nữa. Replay đã được push mọi episode trong run().
        Hàm này chỉ xử lý slow structural update.
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

        # Pre-compute H-step returns for all timesteps so we can pass
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

        # ---- Hai cò súng độc lập cho evaluate_drift() (thay cho
        # record_structural_residual() dùng residual tự nhiễm bẩn ở v1) ----

        # Cò súng 1: probe đóng băng, bịt mắt, đọc trực tiếp proxy.buffer.
        drift_info = self.drift.step(
            episode=int(self.scheduler.episode),
            buffer=self.proxy.buffer,
            n_train_batches=self.cfg.get("drift_train_batches", 5),
        )
        probe_z = float(drift_info.get("z", 0.0) or 0.0)

        # Cò súng 2: nhảy vọt trong ma trận ảnh hưởng có dấu.
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
        # [F1-TOPK] Ground truth cho Core F1 = TOP-K |Phi| ĐO ĐƯỢC, không phải
        # nhãn vai trò tĩnh.
        #
        # gt_core_by_ego / diagnostic_core_by_ego là danh sách vai trò khai
        # báo (collector -> {gatekeeper, relay, blocker, controller}). Sau khi
        # Phi được ĐO từ công thức liên tục (xem _measure_phi_from_sgtp trong
        # omni_arena), hai định nghĩa "core" đã tách hẳn nhau: belief chọn core
        # theo |mu| còn thước đo lại chấm theo nhãn role. Kết quả f1 = 0.000
        # SUỐT RUN là lỗi THƯỚC ĐO, không phải model.
        #
        # Báo cáo CẢ HAI: mean_f1 (vs top-k |Phi| đo được — thước đo chính) và
        # mean_f1_role (vs nhãn role — giữ để có bằng chứng trong paper rằng
        # nhãn role không còn là ground truth hợp lệ sau refactor Phi).
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
                # [ego_conditioned_latent.py] cắm E1/E2 vào CÙNG loss/backward
                # của train_bc -> z_ij thật sự bị ép mang thông tin ego, xem
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

# --- BẮT ĐẦU PATCH RUNNER ---
            import traceback

            # 1. Khởi tạo bộ đếm và Hệ số nhân kiên trì (Persistent Multiplier)
            self._calib_fail_count = getattr(self, '_calib_fail_count', 0)
            self._consecutive_zero_max_p = getattr(self, '_consecutive_zero_max_p', 0)
            self._kappa_multiplier = getattr(self, '_kappa_multiplier', 1.0) 

            # 2. CẬP NHẬT CỜ REACTIVE
            try:
                current_max_p = max([float(np.max(b_mod._p_core_arr)) for b_mod in self.belief_modules.values() if hasattr(b_mod, '_p_core_arr')])
            except:
                current_max_p = 1.0

            # BẮT BUỘC DÙNG 0.05. Nếu dưới 5% cơ hội vào Core, nghĩa là Core đã sập!
            if current_max_p <= 0.05:
                self._consecutive_zero_max_p += 1
            else:
                self._consecutive_zero_max_p = 0

            # 3. Định nghĩa các điều kiện kích hoạt
            is_stage_transition = (stage_before_episode_step == 0 and stage_after_episode_step == 1)
            is_periodic_calib = (stage_after_episode_step == 1 and ep > 0 and (ep % 5 == 0 if ep <= 30 else ep % 25 == 0))
            is_reactive_calib = (self._consecutive_zero_max_p >= 3)
            
            # CẬP NHẬT HỆ SỐ NHÂN (Ghi nhớ tình trạng bệnh)
            if is_reactive_calib:
                # Nếu đang sập: Chém đôi Kappa ngay lập tức, tối thiểu về 0.01
                self._kappa_multiplier = max(0.01, self._kappa_multiplier * 0.5) 
            elif is_periodic_calib and current_max_p > 0.1:
                # Nếu định kỳ chạy mà mô hình đang khỏe (>10%): Phục hồi Kappa dần dần
                self._kappa_multiplier = min(1.0, self._kappa_multiplier * 1.5)

            if is_stage_transition:
                self._reset_switch_counters_if_available()

            # 4. Thực thi Calibration
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
                        sigma_hi = calib["sigma_hi"]  # fallback, chưa đủ mẫu belief thật
                        sigma_p50 = sigma_p90 = sigma_iqr = float("nan")
                    g_anom_mean = float(getattr(self.periph_module, "g_anom_usage_ema", torch.zeros(1)).item())
                    print(f"   [SLOT-DEBUG] sigma_p50={sigma_p50:.6f} sigma_p90={sigma_p90:.6f} "
                          f"sigma_hi={sigma_hi:.6f} g_anom_mean={g_anom_mean:.4f}")
                    print(f"   [VERIFY] sigma_hi={sigma_hi:.6f}  tau_role={tau_role:.6f}")

                    if hasattr(self, 'periph_module'):
                        # [B2.3] truyền CẢ sigma_iqr — trước đây bỏ trống nên
                        # sigma_iqr_floor rơi về đúng sigma_hi và k_sg vẫn nổ.
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
                        
                        # Tính Kappa Gốc và NHÂN VỚI HỆ SỐ KIÊN TRÌ
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
            # --- KẾT THÚC PATCH RUNNER ---

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
                # v2 scheduler không còn EWMA/CUSUM nội bộ (đã thay bằng
                # evaluate_drift() với probe_z/matrix_z) nên hai khoá này
                # không còn trong get_status(); giữ cột lịch sử bằng z-score
                # mới nhất để không phá format lưu trữ hiện có.
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
                # [debug doc mục 1.2] actor/critic/entropy/grad_norm tách
                # riêng -- KHÔNG gộp vào policy_loss nữa, để phân biệt được
                # actor hỏng hay critic hỏng (mục 2.2c).
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