import numpy as np

from envs.causal_adapter import resolve_env_adapter


class MultiEgoReplayBuilder:
    """Build supervised proxy-ensemble data from population trajectories.

    The conditioning set matches the paper:

        r_hat_ij(s_i, a_i, a_j, H_i^{-j}) -> g_i^[0:H], where H is the
        leave-one-out sum of learned per-neighbour embeddings and is
        independent of the learned core/peripheral partition.

    For every time step t, ego i, and neighbor j:
        sample_ij^t = (
            obs_i^t,
            a_i^t,
            a_j^t,
            x_ij^t,
            {x_ik^t : k != i,j},
            [r_i^t, ..., r_i^{t+H-1}]
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

    def build_h_step_returns(self, trajectory, n_agents, complete_only=False):
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
            row, mask_row = {}, {}

            for ego in range(int(n_agents)):
                val = 0.0

                complete = t + self.horizon <= T
                if complete_only and not complete:
                    row[ego] = None
                    continue
                for h in range(self.horizon):
                    if t + h < T:
                        val += (
                            self.discount ** h
                        ) * float(trajectory[t + h]["rewards"][ego])

                row[ego] = float(val)

            out.append(row)

        return out

    def build_lag_rewards(self, trajectory, n_agents, n_horizons=None,
                          return_valid_mask=False):
        """Build direct reward targets at each post-intervention lag.

        ``out[t][ego][ell]`` is ``r_i(t + ell)``.  The response network learns
        these direct lag responses; cumulative responses are derived later as
        ``Q^(h) = sum_{ell<h} gamma**ell * g^[ell]``.  Keeping the primitive
        target non-cumulative makes the latency spectrum identifiable and
        guarantees that an H-step residual is compared with Q at the same H.

        Time-limit boundaries are right-censoring, not zero reward.  The
        returned mask marks observed lags; callers must not treat unobserved
        tail values as terminal outcomes.  An environment can explicitly mark
        a transition as ``terminated`` to state that its future rewards are
        absorbing zeros.
        """
        if n_horizons is None:
            n_horizons = self.horizon

        T = len(trajectory)
        out, masks = [], []

        for t in range(T):
            row, mask_row = {}, {}
            for ego in range(int(n_agents)):
                per_lag = np.zeros((int(n_horizons),), dtype=np.float32)
                valid = np.zeros((int(n_horizons),), dtype=np.float32)
                for lag in range(int(n_horizons)):
                    if t + lag < T:
                        per_lag[lag] = float(trajectory[t + lag]["rewards"][ego])
                        valid[lag] = 1.0
                    elif any(bool(step.get("terminated", False))
                             for step in trajectory[t:]):
                        # An absorbing terminal event, unlike a time limit,
                        # identifies the missing future reward as zero.
                        valid[lag] = 1.0
                row[ego] = per_lag
                mask_row[ego] = valid
            out.append(row)
            masks.append(mask_row)

        return (out, masks) if return_valid_mask else out

    def build_h_step_returns_multi(self, trajectory, n_agents, n_horizons=None):
        """Compatibility helper returning cumulative responses at every H."""
        lag_rewards, lag_masks = self.build_lag_rewards(
            trajectory, n_agents, n_horizons=n_horizons, return_valid_mask=True
        )
        out = []
        for row in lag_rewards:
            cumulative_row = {}
            for ego, values in row.items():
                values = np.asarray(values, dtype=np.float32)
                discounts = np.power(
                    self.discount, np.arange(values.size, dtype=np.float32)
                )
                valid = np.asarray(lag_masks[len(out)][ego], dtype=np.float32)
                cumulative_row[ego] = np.cumsum(values * discounts * valid).astype(np.float32)
            out.append(cumulative_row)
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
        env_adapter = resolve_env_adapter(env)
        h_returns = self.build_h_step_returns(trajectory, n_agents)
        # The network predicts direct lag rewards, not cumulative returns.
        n_horizons_for_push = getattr(proxy_ensemble, "n_horizons", self.horizon)
        lag_rewards, lag_valid_masks = self.build_lag_rewards(
            trajectory, n_agents, n_horizons=n_horizons_for_push,
            return_valid_mask=True,
        )

        pushed = 0

        for t, step in enumerate(trajectory):
            self._require_step_fields(step)
            if (
                int(getattr(proxy_ensemble, "context_item_dim", 0)) > 0
                and "proxy_context_items_excluding" not in step
                and "proxy_context_blocks" not in step
            ):
                raise KeyError(
                    "literal DeepSets proxy requires action-time "
                    "proxy_context_items_excluding or proxy_context_blocks"
                )

            obs_all = step["obs_all"]
            actions = step["actions"]

            # [intervention.py — DR] forced_mask/behaviour_probs exist only in
            # final_runner.py; baseline_runner.py does not use epsilon forcing.
            # ``get`` preserves compatibility with trajectories lacking these
            # fields. The proxy then falls back from DR to the plug-in estimator
            # for that sample via behaviour_prob_j=None, as designed.
            forced_mask = step.get("forced_mask")
            behaviour_probs = step.get("behaviour_probs")
            execution_records = step.get("action_execution_records")
            records_by_agent = {}
            if execution_records:
                for record in execution_records:
                    try:
                        agent_id = int(record.agent_id)
                    except (AttributeError, TypeError, ValueError) as exc:
                        raise ValueError(
                            "action_execution_records must contain typed action records"
                        ) from exc
                    if agent_id in records_by_agent or not 0 <= agent_id < n_agents:
                        raise ValueError("action execution record agent IDs are malformed")
                    records_by_agent[agent_id] = record
                if len(records_by_agent) != n_agents:
                    raise ValueError(
                        "typed action execution records must cover every agent"
                    )

            for ego in range(n_agents):
                obs_i = env_adapter.observation(obs_all, ego)
                action_i = int(actions[ego])
                context_block = step.get("proxy_context_blocks", {}).get(ego)
                if context_block is None:
                    raise KeyError(
                        "candidate-restricted proxy replay requires a ContextBlock"
                    )
                candidate_ids = [
                    int(j) for j in np.asarray(
                        context_block.get("neighbor_ids", ()), dtype=np.int64
                    ).reshape(-1)
                ]
                if len(candidate_ids) != len(set(candidate_ids)) or ego in candidate_ids:
                    raise ValueError("ContextBlock contains invalid candidate neighbour IDs")

                for j in candidate_ids:

                    record = records_by_agent.get(int(j))
                    if execution_records and record is None:
                        raise RuntimeError("missing typed action execution record")

                    # [FIX-X1] x_ij at the correct time t comes from the geometry
                    # snapshot captured by the runner. env.positions now holds
                    # the final episode state and cannot be used here.
                    geom = step.get("geom_snapshot")
                    if geom is None:
                        pair_feat = None
                    else:
                        pair_feat = env_adapter.pair_features_from_snapshot(
                            geom, ego, j
                        )

                    try:
                        z_ex, m_ex = step["proxy_context_excluding"][ego][j]
                    except (KeyError, TypeError, ValueError):
                        raise KeyError(
                            "Proxy replay requires partition-independent "
                            "proxy_context_excluding[ego][neighbor]."
                        )
                    belief_summary = step["belief_summary_cache"][ego]
                    # New replay stores one raw ContextBlock per ego/timestep,
                    # not a copied leave-one-out table per (ego, target) pair.
                    # This is the sum-minus-one DeepSets representation and
                    # reduces collection storage from O(N^3) to O(N^2).
                    context_target_id = int(j)
                    try:
                        context_items, context_mask = step[
                            "proxy_context_items_excluding"
                        ][ego][j]
                    except (KeyError, TypeError, ValueError):
                        context_items, context_mask = None, None
                    target_lags = lag_rewards[t][ego]
                    target_lag_mask = lag_valid_masks[t][ego]
                    horizon_complete = bool(np.all(target_lag_mask > 0.5))
                    # H-step returns are only identified on complete windows.
                    # A time-limit tail is right-censored; keep a finite
                    # placeholder for legacy storage but mark the row via
                    # horizon_complete/target_lag_valid_mask so no H-step
                    # estimator consumes it.  Compute the completeness flag
                    # before reading it; this path must remain valid for the
                    # final runner's first and last trajectory transitions.
                    target_h = (
                        h_returns[t][ego]
                        if horizon_complete and h_returns[t] is not None
                        else 0.0
                    )

                    if record is not None:
                        valid_mask_j = np.asarray(
                            record.valid_action_mask, dtype=bool
                        ).reshape(-1)
                        target_pi_j = np.asarray(
                            record.target_policy_probs, dtype=np.float32
                        ).reshape(-1)
                        q_j = np.asarray(
                            record.reference_probs, dtype=np.float32
                        ).reshape(-1)
                        b_j = np.asarray(
                            record.behavior_probs, dtype=np.float32
                        ).reshape(-1)
                        proposed_action_j = int(record.proposed_action)
                        executed_action_j = int(record.executed_action)
                        target_epsilon_j = float(record.epsilon_used)
                        was_forced = bool(record.was_forced)
                        if executed_action_j != int(actions[j]):
                            raise ValueError(
                                "typed action record disagrees with trajectory executed action"
                            )
                        if target_pi_j.shape != (int(proxy_ensemble.action_dim),) \
                                or q_j.shape != target_pi_j.shape \
                                or b_j.shape != target_pi_j.shape:
                            raise ValueError("typed action policy vectors have wrong shape")
                        behaviour_prob_j = float(b_j[executed_action_j])
                    else:
                        was_forced = False
                        behaviour_prob_j = None
                        if forced_mask is not None:
                            was_forced = bool(forced_mask[j])
                        if behaviour_probs is not None:
                            # b_j(a_j_obs | s) is the effective propensity including
                            # forcing, not raw policy probabilities.
                            behaviour_prob_j = float(
                                behaviour_probs[j][int(actions[j])]
                            )
                        valid_mask_j = (
                            None
                            if step.get("valid_action_masks") is None
                            else np.asarray(step["valid_action_masks"][j], dtype=bool)
                        )
                        target_pi_j = (
                            None if step.get("policy_probs") is None
                            else step["policy_probs"][j]
                        )
                        q_j = None
                        b_j = None
                        proposed_action_j = int(
                            step.get("pre_forcing_actions", actions)[j]
                        )
                        executed_action_j = int(actions[j])
                        target_epsilon_j = None
                    if valid_mask_j is not None:
                        if valid_mask_j.shape != (int(proxy_ensemble.action_dim),) or not np.any(valid_mask_j):
                            raise ValueError(
                                "trajectory valid_action_masks contains an invalid "
                                f"row for neighbour={j}"
                            )
                        if q_j is None:
                            q_j = valid_mask_j.astype(np.float32) / float(valid_mask_j.sum())

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
                        observed_action_j=executed_action_j,
                        z_core_excl_j=z_ex,
                        m_periph_excl_j=m_ex,
                        belief_summary=belief_summary,
                        target_return_h=float(target_h),
                        target_lag_rewards=target_lags,
                        target_lag_valid_mask=target_lag_mask,
                        horizon_complete=horizon_complete,
                        behaviour_prob_j=behaviour_prob_j,
                        was_forced=was_forced,
                        pair_feat=pair_feat,   # [FIX-X1]
                        policy_probs_j=target_pi_j,
                        valid_action_mask=valid_mask_j,
                        episode_id=step.get("episode_id"),
                        timestep=step.get("timestep", t),
                        context_items=context_items,
                        context_mask=context_mask,
                        context_block=context_block,
                        context_target_id=context_target_id,
                        target_action_proposed=proposed_action_j,
                        target_action_executed=executed_action_j,
                        target_pi=target_pi_j,
                        # q_j(a|x) is the fixed reference rule: uniform over
                        # the actions valid at this state.  Store the actual
                        # probability vector, not the boolean mask.
                        target_q=q_j,
                        target_b=(
                            b_j
                        ),
                        target_epsilon=target_epsilon_j,
                        structure_regime_id=int(step.get("structure_regime_id", 0)),
                        policy_version=int(step.get("policy_version", 0)),
                    )

                    pushed += 1

        return int(pushed)
