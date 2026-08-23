"""
intervention.py — ε-forced action controller for CIG-AMF v2.

=============================================================================
WHY THIS FILE EXISTS
=============================================================================
Version 1 defines w_ij with the do-operator (Equations 3–4 in the paper), but
estimates it with a purely observational reward model (Equation 7). The
problem is that a_j is NOT randomized: it is correlated with s, a_i, and the
actions of every other neighbour. The difference between two model
predictions can therefore be correlation disguised as causation, forcing the
paper to concede that it does not claim w_hat = w.

This file fixes the problem at its source. With probability epsilon, agent j
is FORCED to choose a uniformly random action instead of following its policy.
Once a_j is genuinely randomized, every confounding path is cut
MECHANICALLY. This is a literal do(a_j), not an approximation: the equivalent
of a randomized controlled trial that would cost millions in medicine, but is
free here because the simulator is under experimental control.

=============================================================================
IMPORTANT CONSEQUENCE: PROPENSITY IS EXACT
=============================================================================
Before the forcing indicator is realised, the marginal behaviour policy of
agent j at step t is the mixture

    b_j(a | s) = eps / |A_valid(s)| + (1 - eps) * pi_j(a | s)

for valid actions; invalid actions retain probability zero.

b_j is KNOWN EXACTLY because pi_j is the learner's own network and eps is a
chosen constant. Propensity normally has to be estimated in off-policy
evaluation and is a major source of error. Here it is exact, so the augmented
inverse-propensity term can use the logged data-generating probability. This
does not by itself make a row-level conditional effect doubly robust; that
requires an orthogonal second-stage learner or aggregation over repeated
contexts.

Epsilon forcing also guarantees the POSITIVITY/OVERLAP assumption:

    b_j(a | s) >= eps / |A_valid(s)| > 0  for every valid a.

Without this guarantee, a counterfactual action a'_j that j never takes would
force the model to extrapolate into a region with no data, making the estimate
meaningless. Version 1 did not state this assumption anywhere.
=============================================================================
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class EpsilonForcedActionController:
    """
    Manage sparse random action forcing during training.

    Runner usage, replacing the current policy-action sampling site:

        forced_mask, effective_probs = controller.apply(
            actions=actions,             # policy actions, list[int], length n_agents
            policy_probs=policy_probs,   # np [n_agents, action_dim], softmax probabilities
        )
        # Forced positions in actions have been modified in place.
        obs, rew, done, info = env.step(actions)

    When pushing the transition to replay:
        step["forced_mask"]     = forced_mask       # np bool [n_agents]
        step["behaviour_probs"] = effective_probs   # np [n_agents, action_dim]

    Args:
        n_agents:
            Number of agents in the population.
        action_dim:
            Size of the discrete action space.
        eps:
            Per-step forcing probability for each agent. Values in 0.02–0.05
            accumulate enough intervention samples while remaining small
            enough not to destroy reward.
        max_forced_per_step:
            Maximum number of agents forced at once. Simultaneously forcing
            many agents causes mutual interference and complicates credit
            assignment, so this can be capped. None means no cap.
        anneal_to:
            If not None, linearly reduce eps to anneal_to over
            anneal_episodes. This supports stronger early intervention, when
            the proxy needs data, followed by lighter intervention to protect
            final reward.
        anneal_episodes:
            Number of episodes over which to anneal.
        rng:
            Seedable np.random.RandomState for reproducibility.
    """

    def __init__(
        self,
        n_agents: int,
        action_dim: int,
        eps: float = 0.03,
        max_forced_per_step: Optional[int] = None,   # [FIX-P1] was 2
        anneal_to: Optional[float] = None,
        anneal_episodes: int = 60,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)

        self.eps_initial = float(eps)
        self.eps = float(eps)

        # [FIX-P1] None means no cap and is the new default. Cast non-None
        # values to int so a None returned by `cfg.get(..., 2)` is preserved.
        self.max_forced_per_step = (
            None if max_forced_per_step is None else int(max_forced_per_step)
        )
        self.anneal_to = anneal_to
        self.anneal_episodes = int(max(1, anneal_episodes))

        self.rng = rng if rng is not None else np.random.RandomState(0)

        self.episode = 0

        # Per-agent eps; None uses the common eps for all agents.
        self._eps_per_agent = None

        # Paper-reporting statistics, including the reward cost of epsilon.
        self.total_steps = 0
        self.total_forced = 0

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    def step_episode(self):
        """Call once after each episode to update the annealing schedule."""
        self.episode += 1

        if self.anneal_to is None:
            return

        frac = min(1.0, float(self.episode) / float(self.anneal_episodes))
        self.eps = float(
            (1.0 - frac) * self.eps_initial + frac * float(self.anneal_to)
        )

    def get_eps(self) -> float:
        return float(self.eps)

    # ------------------------------------------------------------------
    # Apply interventions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # UNCERTAINTY-TARGETED INTERVENTIONS
    # ------------------------------------------------------------------

    def set_priority(
        self,
        scores: Optional[np.ndarray],
        floor_ratio: float = 0.34,
    ):
        """
        Allocate the intervention budget according to how poorly each agent
        is understood.

        MOTIVATION
        ----------
        The system pays 3% of reward to acquire causal data. Under a uniform
        allocation, most of that 3% is spent on pairs that are already well
        understood, consuming budget without purchasing new information.

        An equivalent example is having three expert-consultation vouchers.
        Spending them on three questions whose answers are already known is
        wasteful; they should be spent on the three most ambiguous questions.

        Here, "most ambiguous" means high epistemic uncertainty: the ensemble
        disagrees most strongly about that agent.

        NON-NEGOTIABLE CONSTRAINT
        -------------------------
        Every agent MUST retain a forcing probability greater than zero. If
        an agent is never forced, overlap assumption A2 fails for that agent
        and all related causal estimates lose their foundation. `floor_ratio`
        therefore guarantees each agent at least that fraction of the mean
        intervention budget.

        CAUSAL VALIDITY IS PRESERVED
        ----------------------------
        The choice of WHOM to force may depend on anything, including history
        and uncertainty, without invalidating causality, provided that:

          (a) after the target is selected, its action remains uniformly
              random and state-independent, preserving do(a_j); and
          (b) each agent's forcing probability is RECORDED and included in
              the propensity, preserving the validity of the DR estimator.

        Both conditions are enforced below: eps_per_agent is incorporated
        into the returned effective_probs.

        Args:
            scores: np [n_agents]; higher values receive higher forcing
                priority. None disables targeting.
            floor_ratio: Budget floor in (0, 1].
        """
        if scores is None:
            self._eps_per_agent = None
            return

        s = np.asarray(scores, dtype=np.float64).reshape(-1)

        if s.shape[0] != self.n_agents:
            raise ValueError(
                f"scores must have length {self.n_agents}; received {s.shape[0]}"
            )

        s = np.clip(s, 0.0, None)
        total = float(s.sum())

        if total <= 1e-12:
            self._eps_per_agent = None
            return

        # Normalize to weights with mean 1.0, then mix in the floor.
        w = s / (total / self.n_agents)
        fr = float(np.clip(floor_ratio, 1e-3, 1.0))
        w = fr + (1.0 - fr) * w

        # Keep the TOTAL BUDGET unchanged: mean(eps_j) = eps.
        w = w / float(np.mean(w))

        eps_j = np.clip(self.eps * w, 0.0, 1.0)

        self._eps_per_agent = eps_j.astype(np.float64)

    def get_eps_per_agent(self) -> np.ndarray:
        """Return the current forcing probability of every agent."""
        if getattr(self, "_eps_per_agent", None) is None:
            return np.full(self.n_agents, self.eps, dtype=np.float64)

        return self._eps_per_agent.copy()

    def apply(
        self,
        actions: List[int],
        policy_probs: np.ndarray,
        valid_action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly force a subset of agents and return effective propensities.

        Args:
            actions:
                Policy-sampled actions as list[int] of length n_agents.
                Forced positions WILL BE MODIFIED IN PLACE.
            policy_probs:
                np.ndarray shape [n_agents, action_dim].
                pi_j(a | s) for every agent and every action.
            valid_action_masks:
                Optional boolean matrix with the same shape. Forcing and the
                logged propensity are normalized over valid actions only.

        Returns:
            forced_mask:
                Boolean array of shape [n_agents]. True means the agent was
                forced.
            effective_probs:
                np.ndarray float32 shape [n_agents, action_dim].
                b_j(a | s), the effective behaviour policy used by the DR
                estimator.
        """
        probs = np.asarray(policy_probs, dtype=np.float32)

        if probs.shape != (self.n_agents, self.action_dim):
            raise ValueError(
                f"policy_probs must have shape [{self.n_agents}, {self.action_dim}]; "
                f"received {probs.shape}"
            )
        if valid_action_masks is None:
            valid = np.ones_like(probs, dtype=bool)
        else:
            valid = np.asarray(valid_action_masks, dtype=bool)
            if valid.shape != probs.shape or np.any(valid.sum(axis=1) == 0):
                raise ValueError(
                    "valid_action_masks must match policy_probs and retain at "
                    "least one action per agent"
                )
        probs = np.where(valid, np.clip(probs, 0.0, None), 0.0)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)

        # ---- 1. Decide which agents are forced ---------------------------
        # eps_vec is uniform unless set_priority enables uncertainty-based
        # targeting, in which case its entries differ by agent.
        eps_vec = self.get_eps_per_agent()          # [n_agents]

        draw = self.rng.rand(self.n_agents)         # [n_agents]
        forced_mask = draw < eps_vec                # [n_agents] bool

        # ------------------------------------------------------------------
        # [FIX-P1] Capping the number of simultaneously forced agents BREAKS
        # the paper's central "propensity known exactly" claim (§B3: b_j "is known exactly,
        # since pi_j is the learner's own network and eps is a chosen
        # constant").
        #
        # The cap is a FILTER THAT DEPENDS ON THE OUTCOME of the population's
        # complete random draw. The probability that j is actually forced is
        # no longer eps_j, but
        #     eps_eff_j = eps_j * P(j survives the cap),
        # where P(...) depends on how many other agents were also selected at
        # that step. With eps=0.03, n=24, and cap=2, X ~ Bin(24, 0.03) and
        # P(X>2) ~ 3.8%, so eps_eff ~ 0.0282. Recorded propensity is therefore
        # systematically overstated by ~6%, and DR divides by exactly the
        # wrong factor, losing unbiasedness.
        #
        # The cap is almost useless under the default regime:
        # E[X] = 24*0.03 = 0.72 agents per step, so cap=2 is almost never hit.
        # It sacrifices the validity of the claim for almost no benefit.
        #
        # Therefore the cap is disabled by default
        # (max_forced_per_step=None). If it is re-enabled, warn once that the
        # recorded propensity is only approximate.
        # ------------------------------------------------------------------
        if self.max_forced_per_step is not None:
            forced_ids = np.flatnonzero(forced_mask)

            if len(forced_ids) > int(self.max_forced_per_step):
                if not getattr(self, "_cap_warned", False):
                    print(
                        "[eps-forcing][WARN] max_forced_per_step="
                        f"{self.max_forced_per_step} truncated the selected agents. "
                        "Recorded nominal-eps propensity now EXCEEDS the actual "
                        "forcing probability, systematically biasing DR and "
                        "invalidating the paper's 'b_j known exactly' claim. "
                        "Set forcer_max_forced_per_step=None to restore validity."
                    )
                    self._cap_warned = True

                keep = self.rng.choice(
                    forced_ids,
                    size=int(self.max_forced_per_step),
                    replace=False,
                )
                forced_mask = np.zeros(self.n_agents, dtype=bool)
                forced_mask[keep] = True

        # ---- 2. Force uniformly random actions ---------------------------
        for j in np.flatnonzero(forced_mask):
            candidates = np.flatnonzero(valid[int(j)])
            actions[int(j)] = int(self.rng.choice(candidates))

        # ---- 3. Compute effective propensities ---------------------------
        # The returned value is the MARGINAL propensity before the forcing
        # indicator is realised, so every row receives
        # b = eps * uniform + (1-eps) * pi. Conditional on F=1 the action
        # distribution is pure uniform; conditional on F=0 it is pi. Mixing
        # those two notions after observing F would yield the wrong weight.
        uniform = valid.astype(np.float32)
        uniform = uniform / uniform.sum(axis=1, keepdims=True)

        # Each agent has its own eps and hence its own propensity. Record the
        # eps actually used; otherwise DR divides by the wrong quantity and
        # loses unbiasedness.
        e = eps_vec.reshape(-1, 1).astype(np.float32)   # [n_agents, 1]

        effective_probs = (
            e * uniform + (1.0 - e) * probs
        ).astype(np.float32)  # [n_agents, action_dim]

        # Renormalize to guard against floating-point error.
        row_sum = np.sum(effective_probs, axis=1, keepdims=True)  # [n_agents, 1]
        effective_probs = effective_probs / np.clip(row_sum, 1e-8, None)

        # ---- 4. Statistics ------------------------------------------------
        self.total_steps += self.n_agents
        self.total_forced += int(np.sum(forced_mask))

        return forced_mask, effective_probs

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, float]:
        """Return statistics for inclusion in the paper's result tables."""
        rate = (
            float(self.total_forced) / float(self.total_steps)
            if self.total_steps > 0
            else 0.0
        )

        return {
            "eps_current": float(self.eps),
            "eps_initial": float(self.eps_initial),
            "total_agent_steps": int(self.total_steps),
            "total_forced": int(self.total_forced),
            "realised_forcing_rate": float(rate),
            "episode": int(self.episode),
            "targeting_enabled": bool(
                getattr(self, "_eps_per_agent", None) is not None
            ),
            "eps_min_agent": float(np.min(self.get_eps_per_agent())),
            "eps_max_agent": float(np.max(self.get_eps_per_agent())),
        }

    def state_dict(self) -> Dict:
        """Serialize the complete intervention stream for exact branching."""
        return {
            "n_agents": int(self.n_agents),
            "action_dim": int(self.action_dim),
            "eps_initial": float(self.eps_initial),
            "eps": float(self.eps),
            "max_forced_per_step": self.max_forced_per_step,
            "anneal_to": self.anneal_to,
            "anneal_episodes": int(self.anneal_episodes),
            "episode": int(self.episode),
            "eps_per_agent": (
                None if self._eps_per_agent is None
                else self._eps_per_agent.copy()
            ),
            "total_steps": int(self.total_steps),
            "total_forced": int(self.total_forced),
            "rng_state": self.rng.get_state(),
            "cap_warned": bool(getattr(self, "_cap_warned", False)),
        }

    def load_state_dict(self, state: Dict):
        """Restore a serialized intervention stream with contract checks."""
        if int(state["n_agents"]) != self.n_agents:
            raise ValueError("forcer checkpoint n_agents mismatch")
        if int(state["action_dim"]) != self.action_dim:
            raise ValueError("forcer checkpoint action_dim mismatch")
        self.eps_initial = float(state["eps_initial"])
        self.eps = float(state["eps"])
        self.max_forced_per_step = state["max_forced_per_step"]
        self.anneal_to = state["anneal_to"]
        self.anneal_episodes = int(state["anneal_episodes"])
        self.episode = int(state["episode"])
        values = state.get("eps_per_agent")
        self._eps_per_agent = (
            None if values is None
            else np.asarray(values, dtype=np.float64).copy()
        )
        self.total_steps = int(state["total_steps"])
        self.total_forced = int(state["total_forced"])
        self.rng.set_state(state["rng_state"])
        self._cap_warned = bool(state.get("cap_warned", False))


class OracleInterventionSampler:
    """
    Wrap environment clone/restore operations to obtain truly causal w_ij.

    The environment already provides:
        clone_state(), restore_state(state),
        rollout_from_current_state(forced={j: action}, horizon, discount)

    This supports two uses:
      (1) Exp3 tiny-oracle calibration, comparing the proxy with ground truth.
      (2) Generating a small number of EXACT intervention samples to anchor
          the proxy. Unlike epsilon forcing, this procedure holds the state
          fixed, changes exactly one action, and compares the outcomes
          directly: a genuine level-3 counterfactual.

    ESTIMAND-ALIGNMENT WARNING (a version-1 defect):
        Version 1 computes  mean_a |f(a) - f(a_obs)|  in the proxy (ALWAYS >= 0)
        but computes         mean_a  (R(a) - R_base)   in the oracle (SIGNED).
        These quantities CANNOT agree. A neighbour with symmetric influence,
        helping under one action and harming under another, yields oracle ~ 0
        but a large proxy value. This is why Exp3 could never produce a good
        result. This class exposes BOTH forms so like estimands can be compared.
    """

    def __init__(
        self,
        env,
        horizon: int = 3,
        discount: float = 0.95,
        n_trials: int = 3,
    ):
        self.env = env
        self.horizon = int(horizon)
        self.discount = float(discount)
        self.n_trials = int(max(1, n_trials))

    def signed_effect(
        self,
        ego_id: int,
        neighbor_id: int,
        candidate_actions: Optional[List[int]] = None,
    ) -> float:
        """
        Return signed w_ij, aligned with proxy v2's `signed_oracle_matched`.

            w = mean_a [ R_i(do(a_j = a)) ] - R_i(baseline)

        w > 0: forcing j to act differently makes i BETTER off, so j is
               currently obstructing i.
        w < 0: forcing j to act differently makes i WORSE off, so j is
               currently helping i.

        Returns:
            float
        """
        return float(
            self.env.compute_oracle_influence_from_current_state(
                ego_id=int(ego_id),
                agent_j=int(neighbor_id),
                intervention_action=(
                    None
                    if candidate_actions is None
                    else int(candidate_actions[0])
                ),
                horizon=self.horizon,
                n_trials=self.n_trials,
                discount=self.discount,
            )
        )

    def range_effect(
        self,
        ego_id: int,
        neighbor_id: int,
        candidate_actions: List[int],
    ) -> float:
        """
        Pieroth-style impact (ICML 2024), Definition 5.1:

            U^{j->i} = max_a R_i(do(a_j=a)) - min_a R_i(do(a_j=a))

        This quantity is always >= 0. It is an important CONTROL BASELINE: if
        the signed signature classifies roles better than this unsigned range,
        that is direct evidence for novelty because Pieroth deliberately
        avoids counterfactuals and does not retain a sign.

        Returns:
            float >= 0
        """
        saved = self.env.clone_state()
        returns_per_action = []

        try:
            for action in candidate_actions:
                trial_vals = []

                for _ in range(self.n_trials):
                    self.env.restore_state(saved)
                    out = self.env.rollout_from_current_state(
                        forced={int(neighbor_id): int(action)},
                        horizon=self.horizon,
                        discount=self.discount,
                    )
                    trial_vals.append(float(out[int(ego_id)]))

                returns_per_action.append(float(np.mean(trial_vals)))
        finally:
            self.env.restore_state(saved)

        if len(returns_per_action) == 0:
            return 0.0

        return float(max(returns_per_action) - min(returns_per_action))

    def full_profile(
        self,
        ego_id: int,
        neighbor_id: int,
        candidate_actions: List[int],
    ) -> Dict[str, float]:
        """
        Return the complete ground-truth profile for one pair, used as the
        reference label when evaluating an influence signature.

        Returns:
            Dictionary with the following keys:
                signed   : mean_a R(a) - R_base            (signed)
                range    : max_a R(a) - min_a R(a)         (Pieroth-style, >=0)
                best     : max_a R(a) - R_base             (maximum help j can provide i)
                worst    : min_a R(a) - R_base             (maximum harm j can cause i)
                spread   : std_a R(a)                      (dispersion across actions)
        """
        saved = self.env.clone_state()

        try:
            base_vals = []
            for _ in range(self.n_trials):
                self.env.restore_state(saved)
                out = self.env.rollout_from_current_state(
                    forced=None,
                    horizon=self.horizon,
                    discount=self.discount,
                )
                base_vals.append(float(out[int(ego_id)]))

            base = float(np.mean(base_vals))

            per_action = []
            for action in candidate_actions:
                trial_vals = []
                for _ in range(self.n_trials):
                    self.env.restore_state(saved)
                    out = self.env.rollout_from_current_state(
                        forced={int(neighbor_id): int(action)},
                        horizon=self.horizon,
                        discount=self.discount,
                    )
                    trial_vals.append(float(out[int(ego_id)]))

                per_action.append(float(np.mean(trial_vals)))
        finally:
            self.env.restore_state(saved)

        if len(per_action) == 0:
            return {
                "signed": 0.0,
                "range": 0.0,
                "best": 0.0,
                "worst": 0.0,
                "spread": 0.0,
            }

        arr = np.asarray(per_action, dtype=np.float64)  # [n_candidates]

        return {
            "signed": float(np.mean(arr) - base),
            "range": float(np.max(arr) - np.min(arr)),
            "best": float(np.max(arr) - base),
            "worst": float(np.min(arr) - base),
            "spread": float(np.std(arr)),
        }
