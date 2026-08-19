import time
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
            dist = torch.distributions.Categorical(probs=probs)
            sampled = dist.sample().detach().cpu().numpy()

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
        for ep in range(int(n_episodes)):
            trajectory, episode_reward, runtime = self.collect_episode()
            policy_loss = self.update_policy(trajectory)

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
                    f"[PureMeanField ep {ep:04d}] "
                    f"reward={mean_reward:.3f} "
                    f"policy_loss={policy_loss:.4f} "
                    f"throughput={throughput:.1f}"
                )

        return self.history

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
            policy_loss = self.update_policy(trajectory)
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
            policy_loss = self.update_policy(trajectory)

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
        - NoTwoTimescale

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
            False for NoTwoTimescale. When False:
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
                adaptive_k=self.cfg.get("belief_adaptive_k", False),
                adaptive_k_min=self.cfg.get("belief_adaptive_k_min", 1),
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
            "value_cache": {},
            # [FIX-X1] Match final_runner by capturing geometry at this
            # timestep so replay_builder constructs time-aligned x_ij.
            "geom_snapshot": {
                "positions": [list(p) for p in self.env.positions],
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
                z_batch.append(self._core_context_excluding(ego, j))
                m_batch.append(self._periph_context_excluding(ego, j))
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

            mu_arr, sigma_arr = self.proxy.score_batch(
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
                    )
                    for j in neighbor_ids
                ],
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

    def _should_update_graph_this_episode(self):
        if not self.use_belief:
            return False

        if not self.use_two_timescale:
            return True

        return self.scheduler.should_update_graph()

    def run(self, n_episodes=100, eval_every=10):
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

            if self.use_two_timescale:
                self.scheduler.step_episode()

                stage_after_episode_step = int(self.scheduler.stage)

                if stage_before_episode_step == 0 and stage_after_episode_step == 1:
                    self._reset_switch_counters_if_available()
            else:
                stage_after_episode_step = int(self.scheduler.stage)

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
                    f"[{self.name} ep {ep:04d}] "
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


class NoTwoTimescaleRunner(SharedAblationBase):
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
