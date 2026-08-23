import numpy as np

from models.structural_proxy import build_pair_feat  # [FIX-X1]


class MultiEgoReplayBuilder:
    """Build supervised proxy-ensemble data from population trajectories.

    The conditioning set matches the paper:

        r_hat_ij(s_i, a_i, a_j, H_i^{-j}) -> R_i^(H), where H is a raw
        leave-one-out observable-set summary independent of the learned
        core/peripheral partition.

    For every time step t, ego i, and neighbor j:
        sample_ij^t = (
            obs_i^t,
            a_i^t,
            a_j^t,
            z_{i,C_i \\ {j}}^t,
            m_{i,P_i \\ {j}}^t,
            b_i^t,
            R_i^{(H)}(t)
        )

    Important invariants:
    - Context is not recomputed in this module.
    - Excluding-j context comes from the trajectory cache at action selection.
    - Each proxy sample therefore matches the state/context used by the policy.
    - ``push_trajectory_to_proxy()`` runs every episode, including Stage 0.
    - Stage 0 does not train or update belief, but still fills the proxy buffer.
    """

    def __init__(self, discount=0.95, horizon=8):
        # P2 (Section 2.2): H_causal >= max_latency + 2. Omni-Arena has four
        # latency tiers: blocker h=1, gatekeeper h=2-3, relay h=4-5, and
        # controller h=6+. Hence H >= 8. The default changed from 3 to 8;
        # callers may still supply another horizon explicitly.
        self.discount = float(discount)
        self.horizon = int(horizon)

    def build_h_step_returns(self, trajectory, n_agents):
        """
        R_i^(H)(t) = sum_{h=0}^{H-1} gamma^h r_i(t+h)

        Args:
            trajectory:
                list step dict.
                Every step must contain the ``rewards`` key.
            n_agents:
                Number of agents in the population.

        Return:
            list length T.
            ``out[t][ego]`` is the ego's H-step return at time t.
        """
        T = len(trajectory)
        out = []

        for t in range(T):
            row = {}

            for ego in range(int(n_agents)):
                val = 0.0

                for h in range(self.horizon):
                    if t + h < T:
                        val += (
                            self.discount ** h
                        ) * float(trajectory[t + h]["rewards"][ego])

                row[ego] = float(val)

            out.append(row)

        return out

    def build_h_step_returns_multi(self, trajectory, n_agents, n_horizons=None):
        """
        P2 multi-horizon form of ``build_h_step_returns()``. It returns
        R_i^(1), R_i^(2), ..., R_i^(n_horizons) for each (t, ego). These are
        real ``target_returns_multi`` values rather than a broadcast scalar for
        structural_proxy_v2 (n_horizons=8 under Section 2.2).

        R_i^(h)(t) = sum_{k=0}^{h-1} gamma^k * r_i(t+k)  (h = 1..n_horizons)

        Return:
            list length T. out[t][ego] = np.ndarray shape [n_horizons].
        """
        if n_horizons is None:
            n_horizons = self.horizon

        T = len(trajectory)
        out = []

        for t in range(T):
            row = {}
            for ego in range(int(n_agents)):
                cum = 0.0
                per_h = np.zeros((int(n_horizons),), dtype=np.float32)
                for h in range(int(n_horizons)):
                    if t + h < T:
                        cum += (self.discount ** h) * float(trajectory[t + h]["rewards"][ego])
                    per_h[h] = cum
                row[ego] = per_h
            out.append(row)

        return out

    def _require_step_fields(self, step):
        required = [
            "obs_all",
            "actions",
            "rewards",
            "proxy_context_excluding",
            "belief_summary_cache",
        ]

        missing = [k for k in required if k not in step]

        if len(missing) > 0:
            raise KeyError(
                "MultiEgoReplayBuilder expected trajectory step to contain "
                f"{required}, but missing {missing}. "
                "Runner must cache context at action-selection time."
            )

    def push_trajectory_to_proxy(
        self,
        trajectory,
        proxy_ensemble,
        env,
    ):
        """Push a complete population trajectory into the proxy buffer.

        Required fields in each step:
            obs_all
            actions
            rewards
            proxy_context_excluding

        Return:
            Number of supervised samples pushed.

        Critical requirements:
        - Call this method every episode, including Stage 0.
        - Stage 0 does not train/update belief, but must collect proxy replay.
        - Do not gate collection on ``scheduler.should_update_graph()``.
        """
        if trajectory is None or len(trajectory) == 0:
            return 0

        n_agents = int(env.n_agents)
        h_returns = self.build_h_step_returns(trajectory, n_agents)
        # P2: read real n_horizons-element targets (default 8) from the
        # trajectory instead of letting add_sample() broadcast one scalar to
        # every horizon, which was the less accurate legacy fallback.
        n_horizons_for_push = getattr(proxy_ensemble, "n_horizons", self.horizon)
        h_returns_multi = self.build_h_step_returns_multi(
            trajectory, n_agents, n_horizons=n_horizons_for_push
        )

        pushed = 0

        for t, step in enumerate(trajectory):
            self._require_step_fields(step)

            obs_all = step["obs_all"]
            actions = step["actions"]

            # [intervention.py — DR] forced_mask/behaviour_probs exist only in
            # final_runner.py; baseline_runner.py does not use epsilon forcing.
            # ``get`` preserves compatibility with trajectories lacking these
            # fields. The proxy then falls back from DR to the plug-in estimator
            # for that sample via behaviour_prob_j=None, as designed.
            forced_mask = step.get("forced_mask")
            behaviour_probs = step.get("behaviour_probs")

            for ego in range(n_agents):
                obs_i = env.get_obs_of_ego(obs_all, ego)
                action_i = int(actions[ego])

                for j in range(n_agents):
                    if j == ego:
                        continue

                    # [FIX-X1] x_ij at the correct time t comes from the geometry
                    # snapshot captured by the runner. env.positions now holds
                    # the final episode state and cannot be used here.
                    geom = step.get("geom_snapshot")
                    if geom is None:
                        pair_feat = None
                    else:
                        pair_feat = build_pair_feat(
                            geom["positions"], geom["agent_zone"],
                            geom["grid_size"], geom["n_zones"], ego, j,
                            agent_role=geom.get("agent_role"),
                        )

                    try:
                        z_ex, m_ex = step["proxy_context_excluding"][ego][j]
                    except (KeyError, TypeError, ValueError):
                        raise KeyError(
                            "Proxy replay requires partition-independent "
                            "proxy_context_excluding[ego][neighbor]."
                        )
                    belief_summary = step["belief_summary_cache"][ego]
                    target_h = h_returns[t][ego]
                    target_multi = h_returns_multi[t][ego]

                    was_forced = False
                    behaviour_prob_j = None

                    if forced_mask is not None:
                        was_forced = bool(forced_mask[j])

                    if behaviour_probs is not None:
                        # b_j(a_j_obs | s) is the EFFECTIVE propensity including
                        # forcing, not raw policy_probs. Confusing the two causes
                        # silent systematic bias in the DR estimator.
                        behaviour_prob_j = float(
                            behaviour_probs[j][int(actions[j])]
                        )

                    # [VERIFY-F1b] Measure at the buffer-write boundary. For
                    # was_forced=True samples, a_j must be approximately uniform
                    # over |A|. Uniform runner hist_action_forced with a skew here
                    # identifies trajectory/buffer label misalignment, matching
                    # the hypothesis that action_j stored the pre-override action.
                    if was_forced:
                        _h = getattr(proxy_ensemble, "_vf1b_hist", None)
                        if _h is None:
                            _h = {}
                            proxy_ensemble._vf1b_hist = _h
                        _a = int(actions[j])
                        _h[_a] = _h.get(_a, 0) + 1

                    proxy_ensemble.add_sample(
                        ego_id=ego,
                        neighbor_id=j,
                        obs_i=obs_i,
                        action_i=action_i,
                        observed_action_j=int(actions[j]),
                        z_core_excl_j=z_ex,
                        m_periph_excl_j=m_ex,
                        belief_summary=belief_summary,
                        target_return_h=float(target_h),
                        target_returns_multi=target_multi,
                        behaviour_prob_j=behaviour_prob_j,
                        was_forced=was_forced,
                        pair_feat=pair_feat,   # [FIX-X1]
                    )

                    pushed += 1

        return int(pushed)
