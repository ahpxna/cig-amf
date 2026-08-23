import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.structural_proxy import (
    LocalCounterfactualProxyEnsemble,
    build_pair_feat,   # [FIX-X1]
    PAIR_FEAT_DIM,
)
from models.belief_layer import BayesLightBeliefState
from models.core_behavior import PairRelationalModule
from models.peripheral_memory import PeripheralMultiMemory
from models.belief_summary import BeliefSummaryBuilder
from models.policy_value import PolicyValueNet
from training.scheduler import TwoTimescaleScheduler
from training.replay_builder import MultiEgoReplayBuilder


def _ordered_geometry_snapshot(env, n_agents):
    """Return geometry ordered by agent id for replay construction.

    OmniArena stores ``positions`` and ``agent_zone`` as dictionaries.  Plain
    iteration over those containers yields integer keys, not position vectors;
    indexing by agent id also keeps this helper compatible with list-backed
    environments.
    """
    return {
        "positions": [list(env.positions[int(i)]) for i in range(int(n_agents))],
        "agent_zone": [int(env.agent_zone[int(i)]) for i in range(int(n_agents))],
        "agent_role": (
            [str(env.agent_role[int(i)]) for i in range(int(n_agents))]
            if hasattr(env, "agent_role") else None
        ),
        "grid_size": int(getattr(env, "grid_size", 1)),
        "n_zones": int(getattr(env, "n_zones", 1)),
    }


def _current_behavioral_phase(env):
    """Return the policy-execution phase active in the collected episode."""
    phase_fn = getattr(env, "_behaviour_mode", None)
    if callable(phase_fn):
        return str(phase_fn())
    return str(getattr(env, "mode", "unknown"))


def _scripted_execution_distribution(env, agent_id, action_dim):
    """Read the scripted H2 policy without consuming environment RNG."""
    fn = getattr(env, "scripted_policy_distribution", None)
    if callable(fn):
        values = np.asarray(fn(int(agent_id)), dtype=np.float64).reshape(-1)
        if values.shape == (int(action_dim),) and np.all(np.isfinite(values)):
            values = np.clip(values, 0.0, None)
            if float(values.sum()) > 0.0:
                return (values / values.sum()).astype(np.float32)
    return np.full(int(action_dim), 1.0 / float(action_dim), dtype=np.float32)


def _adapt_executed_policy(env, cfg, learned_probs):
    """Controlled H2 action-policy intervention for non-causal baselines."""
    learned = np.asarray(learned_probs, dtype=np.float32)
    n_agents, action_dim = learned.shape
    learned = np.clip(learned, 0.0, None)
    learned = learned / np.clip(learned.sum(axis=1, keepdims=True), 1e-12, None)
    requested = float(cfg.get("behavioral_adapter_lambda", 0.0))
    active = requested > 0.0 and (
        not bool(cfg.get("behavioral_adapter_only_in_behavioral_drift", True))
        or str(getattr(env, "mode", "")) == "behavioral_drift"
    )
    lam = float(np.clip(requested, 0.0, 1.0)) if active else 0.0
    scripted = np.stack(
        [_scripted_execution_distribution(env, agent, action_dim) for agent in range(n_agents)],
        axis=0,
    )
    target_ids = cfg.get("behavioral_adapter_target_agents")
    target_roles = cfg.get("behavioral_adapter_target_roles")
    target_mask = np.zeros(n_agents, dtype=bool)
    if target_ids is not None:
        for agent in target_ids:
            if 0 <= int(agent) < n_agents:
                target_mask[int(agent)] = True
    elif target_roles is not None:
        allowed_roles = {str(role) for role in target_roles}
        roles = getattr(env, "agent_role", {})
        for agent in range(n_agents):
            try:
                target_mask[agent] = str(roles[agent]) in allowed_roles
            except (KeyError, IndexError, TypeError):
                target_mask[agent] = False
    else:
        target_mask[:] = True
    executed = learned.copy()
    if active:
        executed[target_mask] = (
            (1.0 - lam) * learned[target_mask] + lam * scripted[target_mask]
        )
    executed = executed / np.clip(executed.sum(axis=1, keepdims=True), 1e-12, None)
    eps = 1e-12
    kl = np.sum(executed * (np.log(np.clip(executed, eps, 1.0))
                              - np.log(np.clip(learned, eps, 1.0))), axis=1)
    tv = 0.5 * np.abs(executed - learned).sum(axis=1)
    selected = target_mask if active else np.zeros(n_agents, dtype=bool)
    unselected = ~selected
    return executed.astype(np.float32), {
        "behavioral_adapter_active": int(active),
        "behavioral_adapter_lambda": lam,
        "behavioral_adapter_kl": float(np.mean(kl)),
        "behavioral_adapter_tv": float(np.mean(tv)),
        "behavioral_adapter_target_count": int(selected.sum()),
        "behavioral_adapter_non_target_count": int(unselected.sum()),
        "behavioral_adapter_target_tv": float(np.mean(tv[selected])) if selected.any() else 0.0,
        "behavioral_adapter_non_target_tv": float(np.mean(tv[unselected])) if unselected.any() else 0.0,
    }


class PureMeanFieldRunner:
    """
    Pure Mean Field baseline.

    Baseline definition:
    - Does not use Bayes-light belief.
    - Does not use a learned core/peripheral partition.
    - Does not use pair-specific relational latents as primary information.
    - The policy receives:
          obs_i
          mean one-hot action over all neighbours
    - This baseline scales cheaply but is blind to structural influence.

    Interface:
        run(n_episodes, eval_every) -> history dict

    History keys intentionally mirror Final-CIGAMF so run_experiment.py can
    store both through the same path.
    """

    _log_tag = "PureMeanField"

    def __init__(self, env, cfg, device="cpu"):
        self.env = env
        self.cfg = dict(cfg)
        self.device = device

        self.n_agents = int(env.n_agents)
        self.obs_dim = int(env.get_obs_dim())
        self.action_dim = int(env.get_action_dim())

        self.hidden = int(cfg.get("policy_hidden", 160))
        self.discount = float(cfg.get("discount", 0.95))

        in_dim = self.obs_dim + self.action_dim

        self.policy_value = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        ).to(device)

        self.actor = nn.Linear(self.hidden, self.action_dim).to(device)
        self.critic = nn.Linear(self.hidden, 1).to(device)

        self.optim = torch.optim.Adam(
            list(self.policy_value.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=float(cfg.get("policy_lr", 1e-3)),
        )

        self.history = {
            "episodes": [],
            "mean_reward": [],
            "reward_per_agent": [],
            "policy_loss": [],
            "runtime": [],
            "throughput_agent_steps_per_sec": [],
            "episode_runtime_total": [],
            "throughput_total_agent_steps_per_sec": [],
            "stage": [],
            "mean_f1": [],
            "mean_temporal_var": [],
            "mean_uncertainty": [],
            "mean_core_size": [],
            "mean_core_switches": [],
            "proxy_loss": [],
            "bc_loss": [],
            "triggered": [],
            "trigger_count": [],
            "proxy_buffer_size": [],
            "pushed_proxy_samples": [],
            "promoted": [],
            "demoted": [],
            "mean_mu": [],
            "max_p": [],
        }
        self.episodes_completed = 0
        self.episode_events = []
        self._last_behavioral_phase = None

    def _neighbor_mean_action(self, actions_list, ego):
        vec = np.zeros(self.action_dim, dtype=np.float32)

        count = 0
        for j in range(self.n_agents):
            if j == ego:
                continue
            a = int(actions_list[j])
            if 0 <= a < self.action_dim:
                vec[a] += 1.0
                count += 1

        if count > 0:
            vec /= float(count)

        return vec

    def _forward(self, obs_t, mean_action_t):
        x = torch.cat([obs_t, mean_action_t], dim=-1)
        h = self.policy_value(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def _observe_episode(self, trajectory):
        """Optional post-episode hook for observational baselines."""
        return None

    def _select_actions_population(self, obs_all):
        """
        The neighbours' action mean requires current actions, which are not yet
        available during action selection. Use env.last_actions as the
        mean-field context, the standard online approximation for a pure
        mean-field baseline.
        """
        last_actions = getattr(self.env, "last_actions", [0 for _ in range(self.n_agents)])

        obs_batch = []
        mf_batch = []

        for ego in range(self.n_agents):
            obs_batch.append(self.env.get_obs_of_ego(obs_all, ego))
            mf_batch.append(self._neighbor_mean_action(last_actions, ego))

        obs_t = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
        mf_t = torch.tensor(np.stack(mf_batch), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, values = self._forward(obs_t, mf_t)
            probs = torch.softmax(logits, dim=-1)
            learned_probs = probs.detach().cpu().numpy()

        executed_probs, diagnostics = _adapt_executed_policy(
            self.env, self.cfg, learned_probs
        )
        self._last_execution_policy_probs = executed_probs
        sampled = [
            int(np.random.choice(self.action_dim, p=executed_probs[ego]))
            for ego in range(self.n_agents)
        ]
        self.last_behavioral_adapter_metrics = diagnostics
        actions = [int(a) for a in sampled]
        values_np = values.detach().cpu().numpy().astype(np.float32)

        return actions, values_np, mf_batch

    def collect_episode(self):
        obs_all = self.env.reset()

        done = False
        trajectory = []
        ep_reward = np.zeros(self.n_agents, dtype=np.float32)

        t0 = time.time()

        while not done:
            actions, values_np, mf_batch = self._select_actions_population(obs_all)

            env_snapshot_before_step = self.env.clone_state()
            next_obs_all, rewards, done, info = self.env.step(actions)
            env_snapshot_after_step = self.env.clone_state()

            rewards = np.array(rewards, dtype=np.float32)
            ep_reward += rewards

            trajectory.append(
                {
                    "obs_all": [x.copy() for x in obs_all],
                    "mean_field_context": [x.copy() for x in mf_batch],
                    "actions": list(actions),
                    "rewards": list(rewards),
                    "values": list(values_np),
                    "execution_policy_probs": np.asarray(
                        getattr(self, "_last_execution_policy_probs", []),
                        dtype=np.float32,
                    ),
                    "env_snapshot_before_step": env_snapshot_before_step,
                    "env_snapshot_after_step": env_snapshot_after_step,
                    "info": info,
                }
            )

            obs_all = next_obs_all

        runtime = time.time() - t0
        adapter_rows = [
            np.asarray(step.get("execution_policy_probs"), dtype=np.float64)
            for step in trajectory
            if np.asarray(step.get("execution_policy_probs", [])).shape
            == (self.n_agents, self.action_dim)
        ]
        if adapter_rows:
            expected = np.mean(np.concatenate(adapter_rows, axis=0), axis=0)
            realised = np.asarray(
                [action for step in trajectory for action in step["actions"]],
                dtype=np.int64,
            )
            observed = np.bincount(
                realised, minlength=self.action_dim
            ).astype(np.float64)
            observed /= max(1.0, float(observed.sum()))
            self.last_behavioral_adapter_metrics = {
                **getattr(self, "last_behavioral_adapter_metrics", {}),
                "behavioral_adapter_action_freq_tv": float(
                    0.5 * np.abs(observed - expected).sum()
                ),
            }
        return trajectory, ep_reward, runtime

    def update_policy(self, trajectory):
        T = len(trajectory)

        if T == 0:
            return 0.0

        returns = [[0.0 for _ in range(self.n_agents)] for _ in range(T)]
        R = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(T)):
            R = np.array(trajectory[t]["rewards"], dtype=np.float32) + self.discount * R
            for ego in range(self.n_agents):
                returns[t][ego] = float(R[ego])

        total_loss = 0.0
        count = 0

        for t, step in enumerate(trajectory):
            obs_batch = []
            mf_batch = []
            actions_batch = []
            returns_batch = []

            for ego in range(self.n_agents):
                obs_batch.append(self.env.get_obs_of_ego(step["obs_all"], ego))
                mf_batch.append(step["mean_field_context"][ego])
                actions_batch.append(int(step["actions"][ego]))
                returns_batch.append(float(returns[t][ego]))

            obs_t = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
            mf_t = torch.tensor(np.stack(mf_batch), dtype=torch.float32, device=self.device)
            action_t = torch.tensor(actions_batch, dtype=torch.long, device=self.device)
            ret_t = torch.tensor(returns_batch, dtype=torch.float32, device=self.device)

            logits, value = self._forward(obs_t, mf_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            logp = dist.log_prob(action_t)
            # [docs/CIG-AMF_training_debug_master.md, Section 2.2(b)]
            # Normalize advantage to mean zero and standard deviation one over
            # the n_agents batch at each timestep. The old implementation used
            # raw advantage, tying gradient scale directly to reward scale.
            # ret_t remains on the original scale as the unchanged critic target.
            adv_raw = (ret_t - value).detach()
            adv = (adv_raw - adv_raw.mean()) / (adv_raw.std(unbiased=False) + 1e-8)

            policy_loss = -logp * adv
            value_loss = F.mse_loss(value, ret_t, reduction="none")
            entropy = dist.entropy()

            loss_vec = policy_loss + 0.5 * value_loss - 0.01 * entropy

            total_loss = total_loss + loss_vec.sum()
            count += self.n_agents

        self.optim.zero_grad()
        loss = total_loss / max(1, count)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.policy_value.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            0.5,
        )

        self.optim.step()

        return float(loss.item())

    def run(self, n_episodes=100, eval_every=10):
        n_episodes = int(n_episodes)
        eval_every = int(eval_every)
        if n_episodes < 0:
            raise ValueError("n_episodes must be non-negative")
        if eval_every <= 0:
            raise ValueError("eval_every must be positive")

        for local_ep in range(n_episodes):
            episode_number = int(self.episodes_completed) + 1
            trajectory, episode_reward, runtime = self.collect_episode()
            policy_loss = (
                0.0 if bool(self.cfg.get("freeze_policy_learning", False))
                else self.update_policy(trajectory)
            )
            self._observe_episode(trajectory)

            agent_steps = float(self.n_agents * len(trajectory))
            throughput = agent_steps / max(float(runtime), 1e-9)
            mean_reward = float(np.mean(episode_reward))

            last_info = trajectory[-1].get("info", {}) if trajectory else {}
            structural_shift_magnitude = float(
                last_info.get("delta_phi_frobenius_structural", 0.0) or 0.0
            )
            behavioral_phase = _current_behavioral_phase(getattr(self, "env", None))
            behavioral_shift = int(
                getattr(self, "_last_behavioral_phase", None) is not None
                and behavioral_phase != self._last_behavioral_phase
            )
            self._last_behavioral_phase = behavioral_phase
            adapter_metrics = getattr(self, "last_behavioral_adapter_metrics", {})
            self.episode_events.append({
                "episode": episode_number,
                "triggered": 0,
                "trigger_count": 0,
                "structural_shift": int(structural_shift_magnitude > 0.0),
                "structural_shift_magnitude": structural_shift_magnitude,
                "behavioral_phase": behavioral_phase,
                "behavioral_shift": behavioral_shift,
                "mean_f1": 0.0,
                "mean_reward": mean_reward,
                "behavioral_adapter_active": int(
                    adapter_metrics.get("behavioral_adapter_active", 0)
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
                    adapter_metrics.get("behavioral_adapter_target_count", 0)
                ),
                "behavioral_adapter_non_target_count": int(
                    adapter_metrics.get("behavioral_adapter_non_target_count", 0)
                ),
                "behavioral_adapter_target_tv": float(
                    adapter_metrics.get("behavioral_adapter_target_tv", 0.0)
                ),
                "behavioral_adapter_non_target_tv": float(
                    adapter_metrics.get("behavioral_adapter_non_target_tv", 0.0)
                ),
            })
            self.episodes_completed = episode_number

            should_record = (
                episode_number % eval_every == 0
                or local_ep == n_episodes - 1
            )
            if should_record:
                self.history["episodes"].append(episode_number)
                self.history["mean_reward"].append(mean_reward)
                self.history["reward_per_agent"].append(mean_reward)
                self.history["policy_loss"].append(float(policy_loss))
                self.history["runtime"].append(float(runtime))
                self.history["throughput_agent_steps_per_sec"].append(float(throughput))
                self.history["episode_runtime_total"].append(float(runtime))
                self.history["throughput_total_agent_steps_per_sec"].append(float(throughput))

                self.history["stage"].append(0)
                self.history["mean_f1"].append(0.0)
                self.history["mean_temporal_var"].append(0.0)
                self.history["mean_uncertainty"].append(0.0)
                self.history["mean_core_size"].append(0.0)
                self.history["mean_core_switches"].append(0.0)
                self.history["proxy_loss"].append(0.0)
                self.history["bc_loss"].append(0.0)
                self.history["triggered"].append(0)
                self.history["trigger_count"].append(0)
                self.history["proxy_buffer_size"].append(0)
                self.history["pushed_proxy_samples"].append(0)
                self.history["promoted"].append(0)
                self.history["demoted"].append(0)
                self.history["mean_mu"].append(0.0)
                self.history["max_p"].append(0.0)

                print(
                    f"[{self._log_tag} ep {episode_number:04d}] "
                    f"reward={mean_reward:.3f} "
                    f"policy_loss={policy_loss:.4f} "
                    f"throughput={throughput:.1f}"
                )

        return self.history


class CorrelationMeanFieldRunner(PureMeanFieldRunner):
    """Observational association-weighted mean-field comparator for H2.

    The baseline maintains a non-negative directed association matrix
    ``W[ego, source]``. For every source action category it computes the
    absolute Pearson correlation between the category indicator and the ego's
    contemporaneous reward, then retains the largest supported correlation.
    W therefore measures association strength, not causal effect.

    Defaults use a rolling 20-episode window, require at least 60 environment
    steps overall, and require both sides of an action indicator to contain at
    least five samples. The policy consumes an action histogram weighted by W;
    if a row has no supported association it falls back to uniform mean field.
    Eq. 33 and policy weighting both use the same non-negative W.
    """

    _log_tag = "CorrelationMeanField"

    def __init__(self, env, cfg, device="cpu"):
        super().__init__(env=env, cfg=cfg, device=device)
        window_episodes = int(cfg.get("association_window_episodes", 20))
        max_steps = int(getattr(env, "max_steps", 30))
        self.association_window_steps = max(1, window_episodes * max_steps)
        self.association_min_steps = int(cfg.get("association_min_steps", 60))
        self.association_min_action_support = int(
            cfg.get("association_min_action_support", 5)
        )
        self.association_statistic = (
            "max_over_actions(abs(PearsonCorr(1[action_source=a], reward_ego)))"
        )
        self.association_signed = False
        self._association_actions = deque(maxlen=self.association_window_steps)
        self._association_rewards = deque(maxlen=self.association_window_steps)
        self._association_matrix = np.zeros(
            (self.n_agents, self.n_agents), dtype=np.float64
        )

    def _neighbor_mean_action(self, actions_list, ego):
        weights = np.asarray(self._association_matrix[int(ego)], dtype=np.float64).copy()
        weights[int(ego)] = 0.0
        total = float(np.sum(weights))
        if total <= 1e-12:
            return super()._neighbor_mean_action(actions_list, ego)

        vector = np.zeros(self.action_dim, dtype=np.float32)
        for source in range(self.n_agents):
            if source == int(ego):
                continue
            action = int(actions_list[source])
            if 0 <= action < self.action_dim:
                vector[action] += float(weights[source] / total)
        return vector

    def _observe_episode(self, trajectory):
        for step in trajectory:
            self._association_actions.append(
                np.asarray(step["actions"], dtype=np.int64).copy()
            )
            self._association_rewards.append(
                np.asarray(step["rewards"], dtype=np.float64).copy()
            )
        self._update_association_matrix()

    def _update_association_matrix(self):
        n_steps = len(self._association_actions)
        if n_steps < self.association_min_steps:
            return

        actions = np.stack(tuple(self._association_actions), axis=0)
        rewards = np.stack(tuple(self._association_rewards), axis=0)
        indicators = np.eye(self.action_dim, dtype=np.float64)[actions]
        features = indicators.reshape(n_steps, self.n_agents * self.action_dim)

        feature_counts = np.sum(features, axis=0)
        supported = (
            (feature_counts >= self.association_min_action_support)
            & ((n_steps - feature_counts) >= self.association_min_action_support)
        )
        features = features - np.mean(features, axis=0, keepdims=True)
        rewards = rewards - np.mean(rewards, axis=0, keepdims=True)
        feature_norm = np.linalg.norm(features, axis=0)
        reward_norm = np.linalg.norm(rewards, axis=0)
        denominator = feature_norm[:, None] * reward_norm[None, :]
        correlations = np.zeros_like(denominator, dtype=np.float64)
        valid = (denominator > 1e-12) & supported[:, None]
        numerator = features.T @ rewards
        correlations[valid] = np.abs(numerator[valid] / denominator[valid])
        correlations = correlations.reshape(
            self.n_agents, self.action_dim, self.n_agents
        )

        # axes: [source, action, ego] -> W[ego, source]
        matrix = np.max(correlations, axis=1).T
        np.fill_diagonal(matrix, 0.0)
        self._association_matrix = np.clip(matrix, 0.0, 1.0)

    def get_influence_matrix(self):
        return self._association_matrix.copy()

class OracleCoreRunner:
    """
    Oracle-core baseline for Experiment 0, the Structure Value gate.

    The oracle PROVIDES the core as the top-k neighbours by true |W*| through
    env.compute_oracle_influence_from_current_state, the same function used by
    structure_value_tier0.py and tier1.py. The core is NOT learned from data.
    This is "zero cost" in the paper's precise sense: zero structure-
    identification cost, not zero policy-training cost. The policy still learns
    through RL, but receives one-hot actions from the true core instead of a
    mean-field or random context.

    core_refresh_every refreshes the core every N steps rather than at every
    step because oracle rollouts for every pair are too expensive. This is a
    deliberate approximation that must be disclosed in reported results; it
    is not an ideal instantaneous oracle.
    """

    _log_tag = "OracleCore"

    def __init__(self, env, cfg, device="cpu"):
        self.env = env
        self.cfg = dict(cfg)
        self.device = device

        self.n_agents = int(env.n_agents)
        self.obs_dim = int(env.get_obs_dim())
        self.action_dim = int(env.get_action_dim())
        self.k_core = int(cfg.get("seed_core_top_k", 3))

        self.hidden = int(cfg.get("policy_hidden", 160))
        self.discount = float(cfg.get("discount", 0.95))
        self.oracle_horizon = int(cfg.get("causal_horizon", 8))
        self.oracle_n_trials = int(cfg.get("oracle_n_trials", 1))
        self.core_refresh_every = int(cfg.get("core_refresh_every", 5))

        in_dim = self.obs_dim + self.k_core * self.action_dim

        self.policy_value = nn.Sequential(
            nn.Linear(in_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
        ).to(device)
        self.actor = nn.Linear(self.hidden, self.action_dim).to(device)
        self.critic = nn.Linear(self.hidden, 1).to(device)

        self.optim = torch.optim.Adam(
            list(self.policy_value.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=float(cfg.get("policy_lr", 1e-3)),
        )

        self._cached_core = {ego: [] for ego in range(self.n_agents)}
        self._steps_since_refresh = 10 ** 9  # Force an immediate first refresh.

        self.history = {
            "episodes": [], "mean_reward": [], "reward_per_agent": [],
            "policy_loss": [], "runtime": [], "throughput_agent_steps_per_sec": [],
            "episode_runtime_total": [], "throughput_total_agent_steps_per_sec": [],
            "stage": [], "mean_f1": [], "mean_temporal_var": [], "mean_uncertainty": [],
            "mean_core_size": [], "mean_core_switches": [], "proxy_loss": [],
            "bc_loss": [], "triggered": [], "trigger_count": [], "proxy_buffer_size": [],
            "pushed_proxy_samples": [], "promoted": [], "demoted": [],
            "mean_mu": [], "max_p": [],
        }

    def _refresh_core_if_needed(self):
        """Use true oracle |W*|; RandomCoreRunner overrides this for its control."""
        if self._steps_since_refresh < self.core_refresh_every:
            self._steps_since_refresh += 1
            return

        # [FIX-O1] The old implementation looped over (ego, j), performing
        # 24 x 23 = 552 oracle rollouts PER refresh and approximately 6–7
        # refreshes per episode, making a run take all night. Looping only over
        # j is sufficient: one intervention on j produces W* for EVERY ego at
        # once because env.step() returns a reward VECTOR. This is an exact
        # 24x reduction with NO approximation or loss of accuracy; it only
        # reuses information already produced by the rollout.
        #
        # [FIX-O2] The old implementation swallowed errors with
        # `except Exception: infl[j] = 0.0`. Near the end of an episode, guard B
        # (self.t + horizon <= max_steps) raised for EVERY pair, making all
        # influences zero. sorted() then returned the first three agents in
        # dictionary order, silently turning the "oracle" into an inert baseline
        # without a warning. Failed refreshes now retain the previous core and
        # are counted for reporting.
        saved = self.env.clone_state()
        infl_matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
        n_failed = 0

        for j in range(self.n_agents):
            try:
                profile = self.env.compute_oracle_influence_all_egos_from_current_state(
                    agent_j=j, intervention_action=self.env.STAY,
                    horizon=self.oracle_horizon, n_trials=self.oracle_n_trials,
                )
                for ego in range(self.n_agents):
                    if ego != j:
                        infl_matrix[ego, j] = abs(float(profile[ego]))
            except AssertionError:
                # P2/P4 guard: the rollout crossed an episode or shift boundary.
                n_failed += 1
            except Exception as e:
                n_failed += 1
                if not getattr(self, "_oracle_warned", False):
                    print(f"[OracleCore][WARN] oracle rollout failed ({type(e).__name__}: {e}) "
                          f"-- retaining the old core instead of treating influence as zero.")
                    self._oracle_warned = True
            finally:
                self.env.restore_state(saved)

        self.env.restore_state(saved)
        self._oracle_failed_refreshes = getattr(self, "_oracle_failed_refreshes", 0)

        if n_failed >= self.n_agents:
            # No measurement was possible; RETAIN the old core instead of noise.
            self._oracle_failed_refreshes += 1
            self._steps_since_refresh = 0
            return

        for ego in range(self.n_agents):
            infl = {j: infl_matrix[ego, j] for j in range(self.n_agents) if j != ego}
            self._cached_core[ego] = sorted(
                infl, key=infl.get, reverse=True
            )[: self.k_core]

        self._steps_since_refresh = 0

    def _core_context(self, actions_list, ego):
        vec = np.zeros(self.k_core * self.action_dim, dtype=np.float32)
        for idx, j in enumerate(self._cached_core.get(ego, [])):
            a = int(actions_list[j])
            if 0 <= a < self.action_dim:
                vec[idx * self.action_dim + a] = 1.0
        return vec

    def _forward(self, obs_t, core_t):
        h = self.policy_value(torch.cat([obs_t, core_t], dim=-1))
        return self.actor(h), self.critic(h).squeeze(-1)

    def _select_actions_population(self, obs_all):
        self._refresh_core_if_needed()
        last_actions = getattr(self.env, "last_actions", [0] * self.n_agents)

        obs_batch, core_batch = [], []
        for ego in range(self.n_agents):
            obs_batch.append(self.env.get_obs_of_ego(obs_all, ego))
            core_batch.append(self._core_context(last_actions, ego))

        obs_t = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
        core_t = torch.tensor(np.stack(core_batch), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, values = self._forward(obs_t, core_t)
            dist = torch.distributions.Categorical(probs=torch.softmax(logits, dim=-1))
            sampled = dist.sample().detach().cpu().numpy()

        return [int(a) for a in sampled], values.detach().cpu().numpy().astype(np.float32), core_batch

    def collect_episode(self):
        obs_all = self.env.reset()
        self._steps_since_refresh = 10 ** 9  # Refresh immediately at episode start.
        done, trajectory = False, []
        ep_reward = np.zeros(self.n_agents, dtype=np.float32)
        t0 = time.time()

        while not done:
            actions, values_np, core_batch = self._select_actions_population(obs_all)
            next_obs_all, rewards, done, info = self.env.step(actions)
            rewards = np.array(rewards, dtype=np.float32)
            ep_reward += rewards
            trajectory.append({
                "obs_all": [x.copy() for x in obs_all],
                "core_context": [x.copy() for x in core_batch],
                "actions": list(actions), "rewards": list(rewards), "values": list(values_np),
            })
            obs_all = next_obs_all

        return trajectory, ep_reward, time.time() - t0

    def update_policy(self, trajectory):
        T = len(trajectory)
        if T == 0:
            return 0.0

        returns = [[0.0] * self.n_agents for _ in range(T)]
        R = np.zeros(self.n_agents, dtype=np.float32)
        for t in reversed(range(T)):
            R = np.array(trajectory[t]["rewards"], dtype=np.float32) + self.discount * R
            for ego in range(self.n_agents):
                returns[t][ego] = float(R[ego])

        total_loss, count = 0.0, 0
        for t, step in enumerate(trajectory):
            obs_batch, core_batch, actions_batch, returns_batch = [], [], [], []
            for ego in range(self.n_agents):
                obs_batch.append(self.env.get_obs_of_ego(step["obs_all"], ego))
                core_batch.append(step["core_context"][ego])
                actions_batch.append(int(step["actions"][ego]))
                returns_batch.append(float(returns[t][ego]))

            obs_t = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
            core_t = torch.tensor(np.stack(core_batch), dtype=torch.float32, device=self.device)
            action_t = torch.tensor(actions_batch, dtype=torch.long, device=self.device)
            ret_t = torch.tensor(returns_batch, dtype=torch.float32, device=self.device)

            logits, value = self._forward(obs_t, core_t)
            dist = torch.distributions.Categorical(probs=torch.softmax(logits, dim=-1))
            logp = dist.log_prob(action_t)
            adv_raw = (ret_t - value).detach()
            adv = (adv_raw - adv_raw.mean()) / (adv_raw.std(unbiased=False) + 1e-8)

            policy_loss = -logp * adv
            value_loss = F.mse_loss(value, ret_t, reduction="none")
            entropy = dist.entropy()
            total_loss = total_loss + (policy_loss + 0.5 * value_loss - 0.01 * entropy).sum()
            count += self.n_agents

        self.optim.zero_grad()
        loss = total_loss / max(1, count)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_value.parameters()) + list(self.actor.parameters())
            + list(self.critic.parameters()), 0.5,
        )
        self.optim.step()
        return float(loss.item())

    def run(self, n_episodes=100, eval_every=10):
        for ep in range(int(n_episodes)):
            trajectory, episode_reward, runtime = self.collect_episode()
            policy_loss = (
                0.0 if bool(self.cfg.get("freeze_policy_learning", False))
                else self.update_policy(trajectory)
            )
            agent_steps = float(self.n_agents * len(trajectory))
            throughput = agent_steps / max(float(runtime), 1e-9)
            mean_reward = float(np.mean(episode_reward))

            if ep % int(eval_every) == 0:
                h = self.history
                h["episodes"].append(ep); h["mean_reward"].append(mean_reward)
                h["reward_per_agent"].append(mean_reward); h["policy_loss"].append(float(policy_loss))
                h["runtime"].append(float(runtime)); h["throughput_agent_steps_per_sec"].append(float(throughput))
                h["episode_runtime_total"].append(float(runtime))
                h["throughput_total_agent_steps_per_sec"].append(float(throughput))
                h["stage"].append(0); h["mean_f1"].append(0.0); h["mean_temporal_var"].append(0.0)
                h["mean_uncertainty"].append(0.0); h["mean_core_size"].append(float(self.k_core))
                h["mean_core_switches"].append(0.0); h["proxy_loss"].append(0.0); h["bc_loss"].append(0.0)
                h["triggered"].append(0); h["trigger_count"].append(0); h["proxy_buffer_size"].append(0)
                h["pushed_proxy_samples"].append(0); h["promoted"].append(0); h["demoted"].append(0)
                h["mean_mu"].append(0.0); h["max_p"].append(0.0)

                print(f"[{self._log_tag} ep {ep:04d}] reward={mean_reward:.3f} "
                      f"policy_loss={policy_loss:.4f} throughput={throughput:.1f}")

        return self.history


class RandomCoreRunner(OracleCoreRunner):
    """
    Experiment 0 control with architecture and input size IDENTICAL to
    OracleCoreRunner, including k_core*action_dim. The ONLY difference is that
    the core is selected RANDOMLY rather than by true |W*|. This is precisely
    the control required for
    learning_range = R[oracle] - R[random] in structure_value_tier2.py.
    """

    _log_tag = "RandomCore"

    def __init__(self, env, cfg, device="cpu"):
        super().__init__(env, cfg, device=device)
        self._rng = np.random.RandomState(int(cfg.get("seed", 0)))

    def _refresh_core_if_needed(self):
        if self._steps_since_refresh < self.core_refresh_every:
            self._steps_since_refresh += 1
            return
        for ego in range(self.n_agents):
            others = [j for j in range(self.n_agents) if j != ego]
            self._rng.shuffle(others)
            self._cached_core[ego] = others[: self.k_core]
        self._steps_since_refresh = 0
class FullExplicitLocalRunner:
    """
    Full Explicit Local baseline.

    Baseline definition:
    - Does not use Bayes-light belief.
    - Does not use a learned core/peripheral partition.
    - Does not use a local counterfactual proxy.
    - Every neighbour in the local population is compressed through explicit
      local-feature pooling.
    - This baseline is richer than mean field but is not structurally filtered.

    Because the environment may not expose a separate observation for each
    neighbour, this implementation uses commonly available fields:
        positions
        agent_zone
        agent_role
        last_actions
        get_obs_of_ego()

    The policy receives:
        obs_i
        explicit_local_summary_i

    explicit_local_summary_i has dimension:
        action_dim + 4
    comprising:
        mean one-hot action over neighbours
        mean normalized relative row
        mean normalized relative col
        mean same-zone indicator
        normalized neighbor count
    """

    def __init__(self, env, cfg, device="cpu"):
        self.env = env
        self.cfg = dict(cfg)
        self.device = device

        self.n_agents = int(env.n_agents)
        self.obs_dim = int(env.get_obs_dim())
        self.action_dim = int(env.get_action_dim())

        self.explicit_dim = self.action_dim + 4
        self.hidden = int(cfg.get("policy_hidden", 160))
        self.discount = float(cfg.get("discount", 0.95))

        in_dim = self.obs_dim + self.explicit_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        ).to(device)

        self.actor = nn.Linear(self.hidden, self.action_dim).to(device)
        self.critic = nn.Linear(self.hidden, 1).to(device)

        self.optim = torch.optim.Adam(
            list(self.backbone.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=float(cfg.get("policy_lr", 1e-3)),
        )

        self.history = {
            "episodes": [],
            "mean_reward": [],
            "reward_per_agent": [],
            "policy_loss": [],
            "runtime": [],
            "throughput_agent_steps_per_sec": [],
            "episode_runtime_total": [],
            "throughput_total_agent_steps_per_sec": [],
            "stage": [],
            "mean_f1": [],
            "mean_temporal_var": [],
            "mean_uncertainty": [],
            "mean_core_size": [],
            "mean_core_switches": [],
            "proxy_loss": [],
            "bc_loss": [],
            "triggered": [],
            "trigger_count": [],
            "proxy_buffer_size": [],
            "pushed_proxy_samples": [],
            "promoted": [],
            "demoted": [],
            "mean_mu": [],
            "max_p": [],
        }

    def _explicit_summary_for_ego(self, ego, actions_ref):
        pi = self.env.positions[int(ego)]
        zi = int(self.env.agent_zone[int(ego)])

        act_mean = np.zeros(self.action_dim, dtype=np.float32)
        rel_rows = []
        rel_cols = []
        same_zones = []
        count = 0

        for j in range(self.n_agents):
            if j == ego:
                continue

            pj = self.env.positions[int(j)]
            zj = int(self.env.agent_zone[int(j)])

            a = int(actions_ref[j]) if j < len(actions_ref) else 0
            if 0 <= a < self.action_dim:
                act_mean[a] += 1.0

            rel_rows.append(float((pj[0] - pi[0]) / max(1, int(self.env.grid_size))))
            rel_cols.append(float((pj[1] - pi[1]) / max(1, int(self.env.grid_size))))
            same_zones.append(1.0 if zi == zj else 0.0)
            count += 1

        if count > 0:
            act_mean /= float(count)

        aux = np.array(
            [
                float(np.mean(rel_rows)) if len(rel_rows) > 0 else 0.0,
                float(np.mean(rel_cols)) if len(rel_cols) > 0 else 0.0,
                float(np.mean(same_zones)) if len(same_zones) > 0 else 0.0,
                float(count / max(1, self.n_agents - 1)),
            ],
            dtype=np.float32,
        )

        return np.concatenate([act_mean, aux], axis=0).astype(np.float32)

    def _forward(self, obs_t, explicit_t):
        x = torch.cat([obs_t, explicit_t], dim=-1)
        h = self.backbone(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def _select_actions_population(self, obs_all):
        last_actions = getattr(self.env, "last_actions", [0 for _ in range(self.n_agents)])

        obs_batch = []
        explicit_batch = []

        for ego in range(self.n_agents):
            obs_batch.append(self.env.get_obs_of_ego(obs_all, ego))
            explicit_batch.append(self._explicit_summary_for_ego(ego, last_actions))

        obs_t = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
        explicit_t = torch.tensor(np.stack(explicit_batch), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, values = self._forward(obs_t, explicit_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            sampled = dist.sample().detach().cpu().numpy()

        actions = [int(a) for a in sampled]
        values_np = values.detach().cpu().numpy().astype(np.float32)

        return actions, values_np, explicit_batch

    def collect_episode(self):
        obs_all = self.env.reset()

        done = False
        trajectory = []
        ep_reward = np.zeros(self.n_agents, dtype=np.float32)

        t0 = time.time()

        while not done:
            actions, values_np, explicit_batch = self._select_actions_population(obs_all)

            env_snapshot_before_step = self.env.clone_state()
            next_obs_all, rewards, done, info = self.env.step(actions)
            env_snapshot_after_step = self.env.clone_state()

            rewards = np.array(rewards, dtype=np.float32)
            ep_reward += rewards

            trajectory.append(
                {
                    "obs_all": [x.copy() for x in obs_all],
                    "explicit_context": [x.copy() for x in explicit_batch],
                    "actions": list(actions),
                    "rewards": list(rewards),
                    "values": list(values_np),
                    "env_snapshot_before_step": env_snapshot_before_step,
                    "env_snapshot_after_step": env_snapshot_after_step,
                    "info": info,
                }
            )

            obs_all = next_obs_all

        runtime = time.time() - t0
        return trajectory, ep_reward, runtime

    def update_policy(self, trajectory):
        T = len(trajectory)

        if T == 0:
            return 0.0

        returns = [[0.0 for _ in range(self.n_agents)] for _ in range(T)]
        R = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(T)):
            R = np.array(trajectory[t]["rewards"], dtype=np.float32) + self.discount * R
            for ego in range(self.n_agents):
                returns[t][ego] = float(R[ego])

        total_loss = 0.0
        count = 0

        for t, step in enumerate(trajectory):
            obs_batch = []
            explicit_batch = []
            actions_batch = []
            returns_batch = []

            for ego in range(self.n_agents):
                obs_batch.append(self.env.get_obs_of_ego(step["obs_all"], ego))
                explicit_batch.append(step["explicit_context"][ego])
                actions_batch.append(int(step["actions"][ego]))
                returns_batch.append(float(returns[t][ego]))

            obs_t = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
            explicit_t = torch.tensor(np.stack(explicit_batch), dtype=torch.float32, device=self.device)
            action_t = torch.tensor(actions_batch, dtype=torch.long, device=self.device)
            ret_t = torch.tensor(returns_batch, dtype=torch.float32, device=self.device)

            logits, value = self._forward(obs_t, explicit_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            logp = dist.log_prob(action_t)
            # [docs/CIG-AMF_training_debug_master.md, Section 2.2(b)]
            # Normalize advantage to mean zero and standard deviation one over
            # the n_agents batch at each timestep. The old implementation used
            # raw advantage, tying gradient scale directly to reward scale.
            # ret_t remains on the original scale as the unchanged critic target.
            adv_raw = (ret_t - value).detach()
            adv = (adv_raw - adv_raw.mean()) / (adv_raw.std(unbiased=False) + 1e-8)

            policy_loss = -logp * adv
            value_loss = F.mse_loss(value, ret_t, reduction="none")
            entropy = dist.entropy()

            loss_vec = policy_loss + 0.5 * value_loss - 0.01 * entropy

            total_loss = total_loss + loss_vec.sum()
            count += self.n_agents

        self.optim.zero_grad()
        loss = total_loss / max(1, count)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            0.5,
        )

        self.optim.step()

        return float(loss.item())

    def run(self, n_episodes=100, eval_every=10):
        for ep in range(int(n_episodes)):
            trajectory, episode_reward, runtime = self.collect_episode()
            policy_loss = (
                0.0 if bool(self.cfg.get("freeze_policy_learning", False))
                else self.update_policy(trajectory)
            )

            agent_steps = float(self.n_agents * len(trajectory))
            throughput = agent_steps / max(float(runtime), 1e-9)
            mean_reward = float(np.mean(episode_reward))

            if ep % int(eval_every) == 0:
                self.history["episodes"].append(ep)
                self.history["mean_reward"].append(mean_reward)
                self.history["reward_per_agent"].append(mean_reward)
                self.history["policy_loss"].append(float(policy_loss))
                self.history["runtime"].append(float(runtime))
                self.history["throughput_agent_steps_per_sec"].append(float(throughput))
                self.history["episode_runtime_total"].append(float(runtime))
                self.history["throughput_total_agent_steps_per_sec"].append(float(throughput))

                self.history["stage"].append(0)
                self.history["mean_f1"].append(0.0)
                self.history["mean_temporal_var"].append(0.0)
                self.history["mean_uncertainty"].append(0.0)
                self.history["mean_core_size"].append(0.0)
                self.history["mean_core_switches"].append(0.0)
                self.history["proxy_loss"].append(0.0)
                self.history["bc_loss"].append(0.0)
                self.history["triggered"].append(0)
                self.history["trigger_count"].append(0)
                self.history["proxy_buffer_size"].append(0)
                self.history["pushed_proxy_samples"].append(0)
                self.history["promoted"].append(0)
                self.history["demoted"].append(0)
                self.history["mean_mu"].append(0.0)
                self.history["max_p"].append(0.0)

                print(
                    f"[FullExplicitLocal ep {ep:04d}] "
                    f"reward={mean_reward:.3f} "
                    f"policy_loss={policy_loss:.4f} "
                    f"throughput={throughput:.1f}"
                )

        return self.history


class SharedAblationBase:
    """
    Shared ablation runner for:
        - NoBelief
        - NoMultiMemory
        - the historical shared-pipeline scheduler approximation

    Unlike PureMeanField and FullExplicitLocal, these ablations retain most of
    the Final-CIGAMF pipeline so each condition tests exactly one module.

    Flags:
        use_belief:
            False for NoBelief. When False:
                - do not train the proxy or belief;
                - do not update the learned core;
                - retain the weak-prior-seeded core partition;
                - continue providing belief_summary to the policy, but derive
                  it from the fixed seeded belief; and
                - exclude the unused proxy's cost from runtime.

        use_multi_memory:
            False for NoMultiMemory. When False:
                - replace the peripheral summary with a single aggregate
                  projection; and
                - do not execute PeripheralMultiMemory.forward.

        use_two_timescale:
            False for the historical scheduler approximation. When False:
                - force the scheduler into Stage 1;
                - update the graph every episode;
                - remove the slow/fast delay; and
                - continue using proxy/belief when use_belief=True.
    """

    def __init__(
        self,
        env,
        cfg,
        device="cpu",
        name="SharedAblation",
        use_belief=True,
        use_multi_memory=True,
        use_two_timescale=True,
    ):
        self.env = env
        self.cfg = dict(cfg)
        self.device = device
        self.name = str(name)

        self.use_belief = bool(use_belief)
        self.use_multi_memory = bool(use_multi_memory)
        self.use_two_timescale = bool(use_two_timescale)

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
            pair_feat_dim=cfg.get("proxy_pair_feat_dim", PAIR_FEAT_DIM),  # [FIX-X1]
            n_ensemble=cfg["n_ensemble"],
            lr=cfg["proxy_lr"],
            buffer_size=cfg.get("proxy_buffer_size", 200000),
            grad_clip=cfg.get("proxy_grad_clip", 1.0),
            device=device,
            # v2 defaults match structural_proxy.py and therefore do not
            # change behaviour when the configuration omits these fields.
            n_horizons=cfg.get("proxy_n_horizons", 3),
            effect_mode=cfg.get("proxy_effect_mode", "signed_aristocrat"),
            use_doubly_robust=cfg.get("proxy_use_doubly_robust", True),
            iw_clip=cfg.get("proxy_iw_clip", 10.0),
            bootstrap_ratio=cfg.get("proxy_bootstrap_ratio", 0.8),
            use_belief_input=cfg.get("proxy_use_belief_input", False),
            ensemble_dropout=cfg.get("proxy_ensemble_dropout", 0.0),
            seed=cfg.get("seed", 0),
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
        ).to(device)

        self.single_periph_proj = nn.Sequential(
            nn.Linear(self.action_dim + 4, self.periph_dim),
            nn.ReLU(),
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

        train_params = (
            list(self.policy_value.parameters())
            + list(self.belief_summary_builder.parameters())
            + list(self.single_periph_proj.parameters())
        )

        if self.use_multi_memory:
            train_params += list(self.periph_module.parameters())

        self.policy_optim = torch.optim.Adam(
            train_params,
            lr=cfg["policy_lr"],
        )

        self.scheduler = TwoTimescaleScheduler(
            k0_warmup=cfg["k0_warmup"],
            alpha_fast=cfg["policy_lr"],
            alpha_slow_ratio=cfg["slow_ratio"],
            accel_factor=cfg["accel_factor"],
            accel_duration=cfg["accel_duration"],
            z_threshold=cfg.get("z_threshold", 3.0),
            require_both=cfg.get("require_both", False),
            refractory=cfg.get("refractory", 10),
            inflation_factor=cfg.get("inflation_factor", 2.5),
            inflation_t_reset=cfg.get("inflation_t_reset", 1),
        )

        if not self.use_two_timescale:
            self.scheduler.force_learned_stage()

        self.replay_builder = MultiEgoReplayBuilder(
            discount=cfg["discount"],
            horizon=cfg["causal_horizon"],
        )

        self.belief_modules = {
            ego: self._make_belief_state(ego)
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
        self.episodes_completed = 0
        self.episode_events = []
        self._last_behavioral_phase = None

    def _make_belief_state(self, ego):
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
                max_core_size=self.cfg.get("max_core_size", 4),
                # v2 defaults match the original recommended values in
                # belief_layer.py and do not change behaviour when unset.
                core_rule=self.cfg.get("belief_core_rule", "lcb"),
                kappa=self.cfg.get("belief_kappa", 1.0),
                alpha_decay=self.cfg.get("belief_alpha_decay", 0.7),
                sigma_alpha_max=self.cfg.get("belief_sigma_alpha_max", 1.0),
                adaptive_k=self.cfg.get("belief_adaptive_k", False),
                adaptive_k_min=self.cfg.get(
                    "belief_adaptive_k_min", self.cfg.get("min_core_size", 1)
                ),
                signed_balance=self.cfg.get("belief_signed_balance", 0.5),
            )
        except TypeError:
            return BayesLightBeliefState(**common_kwargs)

    def _compute_weak_prior_scores(self, ego):
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
        for ego in range(self.n_agents):
            mod = self.belief_modules[ego]
            if hasattr(mod, "reset_switch_counter"):
                mod.reset_switch_counter()

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

    def _single_periph_feature_np(self, ego, exclude_j=None):
        belief_mod = self.belief_modules[ego]
        periph_ids = sorted(list(belief_mod.get_peripheral_set()))
        last_actions = getattr(self.env, "last_actions", [0 for _ in range(self.n_agents)])

        pi = self.env.positions[int(ego)]
        zi = int(self.env.agent_zone[int(ego)])

        act_mean = np.zeros(self.action_dim, dtype=np.float32)
        rel_rows = []
        rel_cols = []
        same_zones = []
        count = 0

        for j in periph_ids:
            if exclude_j is not None and int(j) == int(exclude_j):
                continue

            pj = self.env.positions[int(j)]
            zj = int(self.env.agent_zone[int(j)])

            a = int(last_actions[int(j)]) if int(j) < len(last_actions) else 0
            if 0 <= a < self.action_dim:
                act_mean[a] += 1.0

            rel_rows.append(float((pj[0] - pi[0]) / max(1, int(self.env.grid_size))))
            rel_cols.append(float((pj[1] - pi[1]) / max(1, int(self.env.grid_size))))
            same_zones.append(1.0 if zi == zj else 0.0)
            count += 1

        if count > 0:
            act_mean /= float(count)

        aux = np.array(
            [
                float(np.mean(rel_rows)) if len(rel_rows) > 0 else 0.0,
                float(np.mean(rel_cols)) if len(rel_cols) > 0 else 0.0,
                float(np.mean(same_zones)) if len(same_zones) > 0 else 0.0,
                float(count / max(1, self.n_agents - 1)),
            ],
            dtype=np.float32,
        )

        return np.concatenate([act_mean, aux], axis=0).astype(np.float32)

    def _single_periph_summary_tensor_for_ego(self, ego, exclude_j=None):
        x_np = self._single_periph_feature_np(ego, exclude_j=exclude_j)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.single_periph_proj(x).squeeze(0)

    def _single_periph_summary_np_for_ego(self, ego, exclude_j=None):
        with torch.no_grad():
            return (
                self._single_periph_summary_tensor_for_ego(
                    ego,
                    exclude_j=exclude_j,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

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

    def _periph_summary_tensor_from_inputs(self, inputs, ego=None):
        if self.use_multi_memory:
            return self.periph_module(inputs)

        if ego is None:
            raise ValueError("ego must be provided when use_multi_memory=False")

        return self._single_periph_summary_tensor_for_ego(ego)

    def _periph_summary_np_from_inputs(self, inputs, ego=None):
        with torch.no_grad():
            return (
                self._periph_summary_tensor_from_inputs(inputs, ego=ego)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    def _core_context_excluding(self, ego, exclude_j):
        core_set = self.belief_modules[ego].get_core_set()
        reduced = [x for x in core_set if x != exclude_j]
        return self.pair_rel_module.get_core_summary(ego, reduced)

    @staticmethod
    def _fit_raw_context(values, width):
        """Place a raw leave-one-out summary in a proxy context tensor."""
        out = np.zeros(int(width), dtype=np.float32)
        source = np.asarray(values, dtype=np.float32).reshape(-1)
        out[:min(out.size, source.size)] = source[:out.size]
        return out

    def _raw_proxy_context_excluding(self, ego, exclude_j):
        """Build the proxy's partition-independent leave-one-out context.

        The shared ablations retain the same causal measurement contract as
        the final runner: current observable geometry, zones, and prior
        actions may condition the proxy, while belief state, core membership,
        memory, and pair latents may not.  This prevents an ablation from
        reintroducing the ``proxy -> partition -> proxy`` feedback path.
        """
        ego = int(ego)
        exclude_j = int(exclude_j)
        pos_i = self.env.positions[ego]
        zone_i = int(self.env.agent_zone[ego])
        grid = max(1.0, float(getattr(self.env, "grid_size", 1)))
        zones = max(1.0, float(getattr(self.env, "n_zones", 1) - 1))
        last_actions = getattr(self.env, "last_actions", {})
        rows = []

        for other in range(self.n_agents):
            if other in (ego, exclude_j):
                continue
            pos = self.env.positions[other]
            drow = (float(pos[0]) - float(pos_i[0])) / grid
            dcol = (float(pos[1]) - float(pos_i[1])) / grid
            distance = abs(drow) + abs(dcol)
            zone_delta = (float(self.env.agent_zone[other]) - zone_i) / zones
            same_zone = float(int(self.env.agent_zone[other]) == zone_i)
            try:
                action = float(last_actions[other]) / max(1.0, self.action_dim - 1.0)
            except (KeyError, IndexError, TypeError):
                action = 0.0
            rows.append([drow, dcol, distance, same_zone, zone_delta, action])

        table = np.asarray(rows, dtype=np.float32)
        if table.size == 0:
            summary = np.zeros(18, dtype=np.float32)
        else:
            summary = np.concatenate(
                [table.mean(axis=0), table.std(axis=0), table.max(axis=0)], axis=0
            ).astype(np.float32)

        return (
            self._fit_raw_context(summary, self.core_dim),
            self._fit_raw_context(
                np.concatenate([summary[6:12], summary[:6], summary[12:]]),
                self.periph_dim,
            ),
        )

    def _periph_context_excluding(self, ego, exclude_j):
        if not self.use_multi_memory:
            return self._single_periph_summary_np_for_ego(
                ego,
                exclude_j=exclude_j,
            )

        belief_mod = self.belief_modules[ego]
        periph_ids = sorted(list(belief_mod.get_peripheral_set() - {exclude_j}))
        belief_state = belief_mod.get_state_dict()

        inputs = self.periph_module.build_inputs(
            ego_id=ego,
            peripheral_ids=periph_ids,
            env=self.env,
            belief_state=belief_state,
            prev_core_set=belief_mod.prev_core_set,
        )

        return self._periph_summary_np_from_inputs(inputs, ego=ego)

    def _select_actions_population(self, obs_all):
        actions = {}

        cache = {
            "belief_items_cache": {},
            "belief_summary_cache": {},
            "core_summary_cache": {},
            "periph_inputs_cache": {},
            "periph_summary_cache": {},
            "core_context_excluding": {},
            "periph_context_excluding": {},
            "proxy_context_excluding": {},
            "value_cache": {},
            # [FIX-X1] Match final_runner by capturing geometry at this
            # timestep so replay_builder constructs time-aligned x_ij.
            "geom_snapshot": _ordered_geometry_snapshot(
                self.env, self.n_agents
            ),
        }

        obs_batch = []
        core_batch = []
        periph_batch = []
        belief_batch = []

        for ego in range(self.n_agents):
            belief_items = self._build_belief_items_for_ego(ego)
            belief_summary_np = self._belief_summary_np_from_items(belief_items)
            core_summary_np = self._core_summary_for_ego(ego)

            if self.use_multi_memory:
                periph_inputs = self._build_periph_inputs_for_ego(ego)
                periph_summary_np = self._periph_summary_np_from_inputs(
                    periph_inputs,
                    ego=ego,
                )
            else:
                periph_inputs = None
                periph_summary_np = self._single_periph_summary_np_for_ego(ego)

            cache["belief_items_cache"][ego] = belief_items
            cache["belief_summary_cache"][ego] = belief_summary_np
            cache["core_summary_cache"][ego] = core_summary_np
            cache["periph_inputs_cache"][ego] = periph_inputs
            cache["periph_summary_cache"][ego] = periph_summary_np

            cache["core_context_excluding"][ego] = {}
            cache["periph_context_excluding"][ego] = {}
            cache["proxy_context_excluding"][ego] = {}

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
                cache["proxy_context_excluding"][ego][j] = self._raw_proxy_context_excluding(
                    ego,
                    j,
                )

            obs_batch.append(self.env.get_obs_of_ego(obs_all, ego))
            core_batch.append(core_summary_np)
            periph_batch.append(periph_summary_np)
            belief_batch.append(belief_summary_np)

        obs_t = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
        core_t = torch.tensor(np.stack(core_batch), dtype=torch.float32, device=self.device)
        periph_t = torch.tensor(np.stack(periph_batch), dtype=torch.float32, device=self.device)
        belief_t = torch.tensor(np.stack(belief_batch), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, values = self.policy_value(obs_t, core_t, periph_t, belief_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            sampled = dist.sample().detach().cpu().numpy()

        for ego in range(self.n_agents):
            actions[ego] = int(sampled[ego])
            cache["value_cache"][ego] = float(values[ego].item())

        return actions, cache

    def collect_episode(self):
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
                    "proxy_context_excluding": cache["proxy_context_excluding"],
                    "value_cache": cache["value_cache"],
                    "geom_snapshot": cache["geom_snapshot"],   # [FIX-X1]
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

    def update_policy(self, trajectory):
        T = len(trajectory)

        if T == 0:
            return 0.0

        returns = [[0.0 for _ in range(self.n_agents)] for _ in range(T)]
        R = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(T)):
            R = np.array(trajectory[t]["rewards"], dtype=np.float32) + self.cfg["discount"] * R
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

                if self.use_multi_memory:
                    periph_tensors.append(
                        self._periph_summary_tensor_from_inputs(
                            step["periph_inputs_cache"][ego],
                            ego=ego,
                        )
                    )
                else:
                    periph_tensors.append(
                        self._single_periph_summary_tensor_for_ego(ego)
                    )

            obs_t = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
            core_t = torch.tensor(np.stack(core_batch), dtype=torch.float32, device=self.device)
            belief_t = torch.stack(belief_tensors, dim=0)
            periph_t = torch.stack(periph_tensors, dim=0)

            logits, value = self.policy_value(obs_t, core_t, periph_t, belief_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            action_t = torch.tensor(actions_batch, dtype=torch.long, device=self.device)
            ret_t = torch.tensor(returns_batch, dtype=torch.float32, device=self.device)

            logp = dist.log_prob(action_t)
            # [docs/CIG-AMF_training_debug_master.md, Section 2.2(b)]
            # Normalize advantage to mean zero and standard deviation one over
            # the n_agents batch at each timestep. The old implementation used
            # raw advantage, tying gradient scale directly to reward scale.
            # ret_t remains on the original scale as the unchanged critic target.
            adv_raw = (ret_t - value).detach()
            adv = (adv_raw - adv_raw.mean()) / (adv_raw.std(unbiased=False) + 1e-8)

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
            list(self.policy_value.parameters()),
            0.5,
        )

        self.policy_optim.step()

        return float(loss.item())

    def push_trajectory_to_proxy_buffer(self, trajectory):
        if not self.use_belief:
            return 0

        pushed = self.replay_builder.push_trajectory_to_proxy(
            trajectory=trajectory,
            proxy_ensemble=self.proxy,
            env=self.env,
        )

        return int(pushed)

    def _score_all_pairs_and_update_beliefs(self, obs_all, actions, observed_returns=None, behaviour_probs=None):
        if not self.use_belief:
            return 0, 0

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
                raw_core, raw_periph = self._raw_proxy_context_excluding(ego, j)
                z_batch.append(raw_core)
                m_batch.append(raw_periph)
                b_batch.append(belief_summary)
                neighbor_ids.append(j)

            # Build optional DR inputs when available: observed_returns for ego
            # and behaviour probability of observed neighbour action.
            observed_returns_batch = None
            behaviour_probs_obs_batch = None

            if observed_returns is not None:
                try:
                    # observed_returns may be dict-like mapping ego->value
                    r = float(observed_returns.get(ego, observed_returns[ego]))
                except Exception:
                    try:
                        r = float(observed_returns[ego])
                    except Exception:
                        r = None

                if r is not None:
                    observed_returns_batch = [r for _ in neighbor_ids]

            if behaviour_probs is not None:
                # behaviour_probs expected to be a matrix-like [n_agents, A]
                behaviour_probs_obs_batch = []
                for j in neighbor_ids:
                    try:
                        bp = float(behaviour_probs[j][int(actions[j])])
                    except Exception:
                        bp = None
                    behaviour_probs_obs_batch.append(bp)

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
                # [FIX-X1] Ablation runners must use the SAME conditioning set
                # as Final-CIGAMF. Otherwise the comparison confounds the effect
                # of x_ij with the effect of the ablated mechanism.
                pair_feat_batch=[
                    build_pair_feat(
                        self.env.positions, self.env.agent_zone,
                        getattr(self.env, "grid_size", 1),
                        getattr(self.env, "n_zones", 1), ego, j,
                        agent_role=getattr(self.env, "agent_role", None),
                    )
                    for j in neighbor_ids
                ],
            )

            # C is the standardized response range and is the only signal
            # allowed to update structural belief/core membership.  D remains
            # a behavioural directional diagnostic for the Paper-A protocol.
            mu_arr = out.get("c_mu", out.get("mu_range", out["mu"]))
            sigma_arr = out.get("c_sigma", out["sigma"])

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
        if not self.use_belief:
            return {
                "proxy_loss": 0.0,
                "proxy_train_residual": 0.0,
                "proxy_holdout_residual": 0.0,
                "triggered": 0,
                "promoted": 0,
                "demoted": 0,
            }

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
        # Precompute H-step returns to provide observed_returns to DR.
        try:
            h_returns = self.replay_builder.build_h_step_returns(trajectory, self.n_agents)
        except Exception:
            h_returns = [None for _ in range(len(trajectory))]

        for idx in step_indices:
            step = trajectory[int(idx)]
            self.env.restore_state(step["env_snapshot_before_step"])
            observed_returns = h_returns[int(idx)] if h_returns is not None else None
            behaviour_probs = step.get("behaviour_probs") if isinstance(step, dict) else None

            p_i, d_i = self._score_all_pairs_and_update_beliefs(
                obs_all=step["obs_all"],
                actions=step["actions"],
                observed_returns=observed_returns,
                behaviour_probs=behaviour_probs,
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
        gt_influence = last_info.get("gt_influence_by_ego", None)

        for ego in range(self.n_agents):
            pred_core = self.belief_modules[ego].get_core_set()

            if diagnostic is None:
                gt_core = set()
            else:
                gt_core = diagnostic[ego]

            # Match FinalCIGAMFRunner's primary F1 definition. The role list is
            # only a diagnostic after continuous SGTP Phi replaced the legacy
            # static table; learned cores are scored against measured top-k
            # |Phi| for both treatment and ablation.
            if gt_influence is not None and ego in gt_influence:
                row = gt_influence[ego]
                if row:
                    # Ground truth is independent of the predicted allocation
                    # budget.  Selecting oracle top-k with |pred_core| lets a
                    # collapsed selector inflate its own F1.
                    k = int(self.cfg.get("ground_truth_core_k", 3))
                    gt_core = set(
                        sorted(
                            row,
                            key=lambda source: abs(float(row[source])),
                            reverse=True,
                        )[:k]
                    )

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

    def _should_update_graph_this_episode(self):
        if not self.use_belief:
            return False

        if not self.use_two_timescale:
            return True

        return self.scheduler.should_update_graph()

    def run(self, n_episodes=100, eval_every=10):
        n_episodes = int(n_episodes)
        eval_every = int(eval_every)
        if n_episodes < 0:
            raise ValueError("n_episodes must be non-negative")
        if eval_every <= 0:
            raise ValueError("eval_every must be positive")

        for local_ep in range(n_episodes):
            episode_number = int(self.episodes_completed) + 1
            stage_before_episode_step = int(self.scheduler.stage)

            trajectory, episode_reward, runtime = self.collect_episode()

            pushed_proxy_samples = self.push_trajectory_to_proxy_buffer(trajectory)

            t_policy = time.time()
            learning_frozen = bool(self.cfg.get("freeze_policy_learning", False))
            policy_loss = 0.0 if learning_frozen else self.update_policy(trajectory)
            policy_runtime = time.time() - t_policy

            t_bc = time.time()
            bc_loss = 0.0 if learning_frozen else self.pair_rel_module.train_bc(
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

            # The NoTwoTimescale ablation ignores scheduler frequency but the
            # scheduler clock must still advance.  Keeping it frozen at zero
            # made every trigger share episode 0 and left the refractory state
            # active forever after the first event.
            self.scheduler.step_episode()
            stage_after_episode_step = int(self.scheduler.stage)

            if (self.use_two_timescale
                    and stage_before_episode_step == 0
                    and stage_after_episode_step == 1):
                self._reset_switch_counters_if_available()

            last_info = trajectory[-1].get("info", {}) if trajectory else {}
            structural_shift_magnitude = float(
                last_info.get("delta_phi_frobenius_structural", 0.0) or 0.0
            )
            behavioral_phase = _current_behavioral_phase(getattr(self, "env", None))
            behavioral_shift = int(
                getattr(self, "_last_behavioral_phase", None) is not None
                and behavioral_phase != self._last_behavioral_phase
            )
            self._last_behavioral_phase = behavioral_phase
            self.episode_events.append({
                "episode": episode_number,
                "triggered": int(graph_info["triggered"]),
                "trigger_count": int(trigger_count_now),
                "structural_shift": int(structural_shift_magnitude > 0.0),
                "structural_shift_magnitude": structural_shift_magnitude,
                "behavioral_phase": behavioral_phase,
                "behavioral_shift": behavioral_shift,
                "mean_f1": float(snapshot["mean_f1"]),
                "mean_reward": float(snapshot["mean_reward"]),
            })
            self.episodes_completed = episode_number

            should_record = (
                episode_number % eval_every == 0
                or local_ep == n_episodes - 1
            )
            if should_record:
                self.history["episodes"].append(episode_number)
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
                self.history["triggered"].append(int(graph_info["triggered"]))
                self.history["trigger_count"].append(int(trigger_count_now))
                self.history["stage"].append(int(stage_now))
                self.history["proxy_buffer_size"].append(
                    int(self.proxy.get_buffer_size()) if self.use_belief else 0
                )
                self.history["pushed_proxy_samples"].append(
                    int(pushed_proxy_samples)
                )
                self.history["promoted"].append(int(graph_info["promoted"]))
                self.history["demoted"].append(int(graph_info["demoted"]))

                print(
                    f"[{self.name} ep {episode_number:04d}] "
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
                    f"buffer={self.proxy.get_buffer_size() if self.use_belief else 0} "
                    f"pushed={pushed_proxy_samples} "
                    f"promoted={graph_info['promoted']} "
                    f"demoted={graph_info['demoted']} "
                    f"proxy_train_res={graph_info.get('proxy_train_residual', 0.0):.5f} "
                    f"proxy_holdout_res={graph_info.get('proxy_holdout_residual', 0.0):.5f} "
                    f"throughput={snapshot['throughput_agent_steps_per_sec']:.1f} "
                    f"throughput_total={snapshot['throughput_total_agent_steps_per_sec']:.1f}"
                )

        return self.history


class NoBeliefRunner(SharedAblationBase):
    def __init__(self, env, cfg, device="cpu"):
        super().__init__(
            env=env,
            cfg=cfg,
            device=device,
            name="NoBelief",
            use_belief=False,
            use_multi_memory=True,
            use_two_timescale=True,
        )


class NoMultiMemoryRunner(SharedAblationBase):
    def __init__(self, env, cfg, device="cpu"):
        super().__init__(
            env=env,
            cfg=cfg,
            device=device,
            name="NoMultiMemory",
            use_belief=True,
            use_multi_memory=False,
            use_two_timescale=True,
        )


class LegacySharedNoTwoTimescaleRunner(SharedAblationBase):
    """Historical approximation retained only for result archaeology.

    This implementation predates Final's forced-action propensity, signature
    tracker, ego-conditioned heads, and auxiliary-loss path. It is therefore
    not a valid scheduler-only ablation and is not registered by the factory.
    """

    def __init__(self, env, cfg, device="cpu"):
        super().__init__(
            env=env,
            cfg=cfg,
            device=device,
            name="NoTwoTimescale",
            use_belief=True,
            use_multi_memory=True,
            use_two_timescale=False,
        )
