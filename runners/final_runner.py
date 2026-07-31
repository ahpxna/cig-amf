import time
import numpy as np
import torch
import torch.nn.functional as F

from models.structural_proxy import LocalCounterfactualProxyEnsemble
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
            mu_floor=0.05,             # Tăng sàn lên 5%
            beta_floor=0.05,
            use_uniform_mix=True,      # ÉP CỨNG BẬT MIX (Rất quan trọng!)
            uniform_mix=0.3,           # Ép cứng 30% nhiễu
            lb_coeff=0.5,              # ÉP CỨNG LỰC PHẠT (Gấp 10 lần mặc định)
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
            z_threshold=cfg.get("z_threshold", 3.0),
            refractory=cfg.get("refractory", 10),
            inflation_factor=cfg.get("inflation_factor", 2.5)
        )
        self.sig_tracker = InfluenceSignatureTracker(n_agents=self.n_agents, window=30)
        self.forcer = EpsilonForcedActionController(
            n_agents=self.n_agents, action_dim=self.action_dim, eps=cfg.get("eps", 0.03),
            max_forced_per_step=2, anneal_to=0.01, anneal_episodes=60,
            rng=np.random.RandomState(cfg.get("seed", 0))
        )
        self.heads = EgoConditionedHeads(latent_dim=self.pair_rel_module.hidden_dim, n_horizons=3).to(self.device)
        self.drift = DriftDetector(obs_dim=self.obs_dim, action_dim=self.action_dim, n_horizons=3,
                                   warmup_batches=200, recalibrate_after=15, seed=cfg.get("seed", 0), device=self.device)
        self.matdet = MatrixDriftDetector(window=20)
        self.recip = ReciprocityTracker(n_agents=self.n_agents, min_causal_samples=20)

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
            )
            for ego in range(self.n_agents)
        }

        self._initialize_seeded_cores()

        self.history = {
            "episodes": [],
            "mean_reward": [],
            "reward_per_agent": [],
            "mean_f1": [],
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

        for ego in range(self.n_agents):
            actions[ego] = int(sampled[ego])
            cache["value_cache"][ego] = float(values_np[ego])

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
                    "belief_summary_cache": cache["belief_summary_cache"],
                    "core_summary_cache": cache["core_summary_cache"],
                    "periph_inputs_cache": cache["periph_inputs_cache"],
                    "periph_summary_cache": cache["periph_summary_cache"],
                    "core_context_excluding": cache["core_context_excluding"],
                    "periph_context_excluding": cache["periph_context_excluding"],
                    "value_cache": cache["value_cache"],
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

            obs_all = next_obs_all

        runtime = time.time() - t0

        return trajectory, ep_reward, runtime

    # ============================================================
    # Policy update
    # ============================================================

    def update_policy(self, trajectory):
        T = len(trajectory)

        if T == 0:
            return 0.0

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

                periph_tensors.append(
                    self._periph_summary_tensor_from_inputs(
                        step["periph_inputs_cache"][ego]
                    )
                )

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
            adv = (ret_t - value).detach()

            policy_loss = -logp * adv
            value_loss = F.mse_loss(value, ret_t, reduction="none")
            entropy = dist.entropy()

            loss_vec = policy_loss + 0.5 * value_loss - 0.01 * entropy

            total_loss = total_loss + loss_vec.sum()
            count += self.n_agents

        self.policy_optim.zero_grad()

        loss = total_loss / max(1, count)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.policy_value.parameters())
            + list(self.periph_module.parameters())
            + list(self.belief_summary_builder.parameters()),
            0.5,
        )

        self.policy_optim.step()

        return float(loss.item())

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

    def _score_all_pairs_and_update_beliefs(self, obs_all, actions):
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

            mu_arr, sigma_arr = self.proxy.score_batch(
                obs_i_batch=obs_i_batch,
                action_i_batch=action_i_batch,
                observed_action_j_batch=action_j_batch,
                z_core_excl_j_batch=z_batch,
                m_periph_excl_j_batch=m_batch,
                belief_summary_batch=b_batch,
            )

            mu_sigma = {
                j: (float(mu_arr[k]), float(sigma_arr[k]))
                for k, j in enumerate(neighbor_ids)
            }

            update_result = self.belief_modules[ego].update_batch(mu_sigma)

            if update_result is None:
                promoted = set()
                demoted = set()
            else:
                promoted, demoted = update_result

            self.pair_rel_module.warm_start_if_promoted(ego, promoted)

            total_promoted += len(promoted)
            total_demoted += len(demoted)

        return int(total_promoted), int(total_demoted)

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

        promoted = 0
        demoted = 0

        for idx in step_indices:
            step = trajectory[int(idx)]
            self.env.restore_state(step["env_snapshot_before_step"])
            p_i, d_i = self._score_all_pairs_and_update_beliefs(
                obs_all=step["obs_all"],
                actions=step["actions"],
            )
            promoted += int(p_i)
            demoted += int(d_i)

        self.env.restore_state(trajectory[-1]["env_snapshot_after_step"])

        residual = self.proxy.get_latest_residual()
        triggered = int(self.scheduler.record_structural_residual(residual))

        return {
            "proxy_loss": float(proxy_loss),
            "proxy_train_residual": float(
                getattr(self.proxy, "get_latest_train_residual", lambda: residual)()
            ),
            "proxy_holdout_residual": float(
                getattr(self.proxy, "get_latest_holdout_residual", lambda: residual)()
            ),
            "triggered": int(triggered),
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

        for ego in range(self.n_agents):
            pred_core = self.belief_modules[ego].get_core_set()

            if diagnostic is None:
                gt_core = set()
            else:
                gt_core = diagnostic[ego]

            f1s.append(self._core_f1(pred_core, gt_core))
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
            policy_loss = self.update_policy(trajectory)
            policy_runtime = time.time() - t_policy

            t_bc = time.time()
            bc_loss = self.pair_rel_module.train_bc(
                n_steps=self.cfg["bc_train_steps"],
                batch_size=self.cfg["bc_batch_size"],
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

            if stage_before_episode_step == 0 and stage_after_episode_step == 1:
                self._reset_switch_counters_if_available()

            if ep % int(eval_every) == 0:
                self.history["episodes"].append(ep)
                self.history["mean_reward"].append(snapshot["mean_reward"])
                self.history["reward_per_agent"].append(snapshot["reward_per_agent"])
                self.history["mean_f1"].append(snapshot["mean_f1"])
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
                self.history["scheduler_residual_ewma"].append(
                    float(sched_status.get("residual_ewma", 0.0) or 0.0)
                )
                self.history["scheduler_cusum_score"].append(
                    float(sched_status.get("cusum_score", 0.0))
                )
                self.history["scheduler_accel_remaining"].append(
                    int(sched_status.get("accel_remaining", 0))
                )
                self.history["proxy_loss"].append(float(graph_info["proxy_loss"]))
                self.history["bc_loss"].append(float(bc_loss))
                self.history["policy_loss"].append(float(policy_loss))
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
                    f"proxy_train_res={graph_info.get('proxy_train_residual', 0.0):.5f} "
                    f"proxy_holdout_res={graph_info.get('proxy_holdout_residual', 0.0):.5f} "
                    f"throughput={snapshot['throughput_agent_steps_per_sec']:.1f} "
                    f"throughput_total={snapshot['throughput_total_agent_steps_per_sec']:.1f}"
                )

        return self.history