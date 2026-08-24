"""
belief_layer_v2.py — corrected Bayes-light structural belief.

=============================================================================
V1 DEFECTS, DIAGNOSED USING THE PAPER'S OWN RESULTS
=============================================================================

[B1] p_core SATURATED BECAUSE sigma WAS IN THE DENOMINATOR.
     v1 used:
         score  = (|mu_bar| - tau) / (sigma_bar + eps)
         p_core = sigmoid(score)

     A small sigma made the denominator small, amplified the sigmoid argument,
     and saturated p_core at zero or one. The paper's own table confirms this:
        NoTwoTimescale: sigma=0.050 meant division by 0.05, equivalent to 20x
                        amplification. Only one neighbour survived and core
                        size was 1.0, exactly as observed.
        Final CIG-AMF:  sigma=0.663 kept sigmoid softer, so more neighbours
                        crossed tau_in until core size reached the observed
                        max_core_size ceiling of 6.0.

     Core size was therefore determined by sigma magnitude rather than by
     influence structure. The causal information in mu was overwhelmed.

     The deeper cause was that sigma was used TWICE: once to regulate the
     learning rate in Eq. 13, which is reasonable, and again as the threshold
     denominator in Eq. 14, which caused saturation. The effects compounded.

     v2 SEPARATES THE TWO ROLES. sigma regulates only learning speed. Core
     selection uses an uncertainty-penalized capacity score:
         score_lcb = |mu_bar| - kappa * sigma_bar
     High uncertainty is SUBTRACTED, making core entry conservatively harder,
     but cannot saturate the score because sigma is additive rather than a
     divisor.

[B2] CORE SIZE WAS A HARD CONSTANT, NOT A LEARNED RESULT.
     v1 used min_core_size/max_core_size. Across five seeds the reported sizes
     were 6.0+-0.0, 1.0+-0.0, and 2.0+-0.0. Exact zero standard deviation is
     nearly impossible for a genuinely learned quantity. All three values were
     boundary outcomes: ceiling saturation, floor collapse, or unchanged seed
     state. None was a learned choice.

     v2 retains min/max as safety valves but:
       - counts and reports boundary-hit rates through get_saturation_stats;
       - adds adaptive_k, which scales k with influence-distribution entropy.
         Influence concentrated on a few neighbours yields small k; diffuse
         influence yields large k. Capacity is then genuinely adaptive.

[B3] alpha VIOLATED ROBBINS-MONRO, PRECLUDING A CONVERGENCE CLAIM.
     v1 used alpha=lambda_0/(1+c*sigma). Its positive lower bound
     lambda_0/(1+c*sigma_max)>0 implies sum(alpha^2)=infinity. Pieroth (ICML
     2024) Theorem 5.6 proves almost-sure convergence for this type of
     iteration under Assumption 3.3(c):
         sum(alpha_t)=infinity AND sum(alpha_t^2)<infinity
     Their implementation uses alpha_t=alpha_0/t^d with d=0.726.

     v2 uses alpha_t=lambda_0/(t^decay*(1+c*sigma)) with decay in (0.5,1].
     This satisfies Robbins-Monro and allows the paper to reuse Pieroth's proof
     framework for an almost-free convergence proposition.

[B4] BELIEF HAD NO SIGN.
     v1 stored mu_bar but applied |mu_bar| everywhere, making harmful agents
     indistinguishable from helpful ones. v2 preserves sign throughout and
     supplies it to the beneficial/harmful semantic slots.
=============================================================================
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class BayesLightBeliefState:
    """
    Structural-capacity belief for ego agent i.

    State per directed pair stores smoothed capacity ``C_bar`` and
    ``sigma_C_bar``.  Direction ``D`` is intentionally not a state variable:
    it belongs to the fast peripheral semantic representation.

    Args:
        core_rule:
            "lcb": recommended default, |mu|-kappa*sigma
            "p_core": legacy p_core threshold for ablation
            "signed": separately balance top helpful and harmful agents
        kappa:
            LCB uncertainty penalty; larger values produce smaller conservative cores.
        alpha_decay:
            Exponent d in alpha~1/t^d. Values in (0.5,1] satisfy
            Robbins-Monro; zero restores v1 behaviour for ablation.
        adaptive_k:
            If true, scale target k with influence-distribution entropy.
    """

    def __init__(
        self,
        ego_id: int,
        neighbor_ids: List[int],
        lambda_0: float = 0.12,
        uncertainty_scale: float = 2.0,
        tau: float = 0.10,
        tau_in: float = 0.62,
        tau_out: float = 0.46,
        weak_prior_top_k: int = 2,
        min_core_size: int = 1,
        max_core_size: int = 4,
        sigma_floor: float = 0.01,
        eps: float = 1e-6,
        # v2 parameters.
        core_rule: str = "lcb",
        kappa: float = 1.0,
        alpha_decay: float = 0.7,
        adaptive_k: bool = False,
        adaptive_k_min: Optional[int] = None,
        signed_balance: float = 0.5,
        sigma_alpha_max: float = 1.0,
    ):
        self.ego_id = int(ego_id)
        self.neighbor_ids = [int(j) for j in neighbor_ids]

        self.lambda_0 = float(lambda_0)
        self.uncertainty_scale = float(uncertainty_scale)
        self.tau = float(tau)
        self.tau_in = float(tau_in)
        self.tau_out = float(tau_out)

        self.weak_prior_top_k = int(weak_prior_top_k)
        self.min_core_size = max(0, int(min_core_size))
        self.max_core_size = int(max_core_size)

        self.sigma_floor = float(sigma_floor)
        self.eps = float(eps)

        if self.max_core_size <= 0:
            self.max_core_size = len(self.neighbor_ids)

        self.max_core_size = min(self.max_core_size, len(self.neighbor_ids))
        self.min_core_size = min(self.min_core_size, self.max_core_size)

        if core_rule not in ("lcb", "p_core", "signed"):
            raise ValueError(f"invalid core_rule: {core_rule}")

        self.core_rule = str(core_rule)
        self.kappa = float(kappa)
        self.alpha_decay = float(alpha_decay)
        self.adaptive_k = bool(adaptive_k)
        # There is one lower capacity bound.  Keeping a second adaptive-only
        # minimum allowed _apply_capacity() to first fill to one value and
        # then prune to another, so ``min_core_size`` was not a true minimum.
        if adaptive_k_min is not None and int(adaptive_k_min) != self.min_core_size:
            raise ValueError(
                "adaptive_k_min must equal min_core_size; the core budget has "
                "a single lower bound"
            )
        self.adaptive_k_min = self.min_core_size
        self.signed_balance = float(np.clip(signed_balance, 0.0, 1.0))
        self.sigma_alpha_max = max(float(sigma_alpha_max), self.sigma_floor)

        # ---------------------------------------------------------------
        # [GPU contract sections 1.2/1.4] Authoritative state lives in float64
        # NumPy arrays. One ego has only a few hundred values, so GPU transfer
        # for synchronization would add cost without benefit. EWMA, debiasing,
        # LCB, hysteresis, and top-k capacity are vectorized over these arrays.
        #
        # Legacy dictionaries remain synchronized views because diagnostics.py
        # reads mod.mu_bar directly. They are bulk-updated once per batch and
        # are not authoritative.
        # ---------------------------------------------------------------
        n = len(self.neighbor_ids)
        self._pos: Dict[int, int] = {j: i for i, j in enumerate(self.neighbor_ids)}

        self._mu_arr = np.zeros(n, dtype=np.float64)
        self._sigma_arr = np.ones(n, dtype=np.float64)
        self._bias_corr_arr = np.ones(n, dtype=np.float64)
        self._mu_init_arr = np.full(n, self.MU_INIT, dtype=np.float64)
        self._sigma_init_arr = np.full(n, self.SIGMA_INIT, dtype=np.float64)
        self._n_updates_arr = np.zeros(n, dtype=np.int64)
        self._p_core_arr = np.full(n, 0.5, dtype=np.float64)

        # Dictionary views preserve direct-read compatibility.
        self.mu_bar: Dict[int, float] = dict(zip(self.neighbor_ids, self._mu_arr.tolist()))
        self.sigma_bar: Dict[int, float] = dict(zip(self.neighbor_ids, self._sigma_arr.tolist()))
        self.p_core: Dict[int, float] = dict(zip(self.neighbor_ids, self._p_core_arr.tolist()))
        self.n_updates: Dict[int, int] = dict(zip(self.neighbor_ids, self._n_updates_arr.tolist()))

        # Retain these dictionaries for legacy dict-of-float access paths.
        self._bias_corr: Dict[int, float] = dict(zip(self.neighbor_ids, self._bias_corr_arr.tolist()))
        self._mu_init: Dict[int, float] = dict(zip(self.neighbor_ids, self._mu_init_arr.tolist()))
        self._sigma_init: Dict[int, float] = dict(zip(self.neighbor_ids, self._sigma_init_arr.tolist()))

        # Count uncertainty inflations for paper reporting.
        self.n_inflations = 0
        self.last_inflation_episode = None

        self.core_set: Set[int] = set()
        self.prev_core_set: Set[int] = set()
        self.seeded_core_set: Set[int] = set()

        self.last_promoted: Set[int] = set()
        self.last_demoted: Set[int] = set()
        self.last_core_switch_count = 0

        self.mu_history: Dict[int, List[float]] = {j: [] for j in self.neighbor_ids}
        self.sigma_history: Dict[int, List[float]] = {j: [] for j in self.neighbor_ids}
        self.p_history: Dict[int, List[float]] = {j: [] for j in self.neighbor_ids}
        self.core_history: List[Set[int]] = []

        # [B2] Count boundary hits; this must be reported in the paper.
        self.n_core_updates = 0
        self.n_hit_max = 0
        self.n_hit_min = 0

    # =====================================================================
    # Helper
    # =====================================================================

    # Initial belief values required for general bias correction.
    MU_INIT = 0.0
    SIGMA_INIT = 1.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    def _safe_sigma(self, sigma) -> float:
        return max(float(sigma), self.sigma_floor, self.eps)

    def debiased_mu(self, j: int) -> float:
        """
        Correct mu_bar for initialization bias.

            mu_hat = mu_bar / (1 - prod(1 - alpha_s))

        Use this for every decision, including core selection and role
        assignment. Raw mu_bar is for logging only.
        """
        j = int(j)
        prod = float(self._bias_corr.get(j, 1.0))
        denom = 1.0 - prod

        mu_init = float(self._mu_init.get(j, self.MU_INIT))

        if denom < 1e-3:
            # Insufficient evidence beyond the anchor. Return the anchor rather
            # than hard-coded zero: startup remains legacy-equivalent at zero,
            # while post-inflation re-anchoring preserves the best estimate.
            return mu_init

        return (float(self.mu_bar[j]) - prod * mu_init) / denom

    def debiased_sigma(self, j: int) -> float:
        """
        Correct sigma_bar for initialization bias. Otherwise residual
        sigma=1.0 prior inflates uncertainty, makes LCB negative, and empties
        the core.
        """
        j = int(j)
        prod = float(self._bias_corr.get(j, 1.0))
        denom = 1.0 - prod

        sigma_init = float(self._sigma_init.get(j, self.SIGMA_INIT))

        if denom < 1e-3:
            return max(sigma_init, self.sigma_floor)

        # ---------------------------------------------------------------
        # General formula; a unit test exposed this defect.
        #
        # Adam uses m/(1-beta^t) because m starts at zero. Sigma starts at 1.0
        # as a maximum-uncertainty prior, so:
        #     sigma_bar = prod*1.0 + (1-prod)*sigma_thuc
        # Dividing directly by (1-prod) adds prod/(1-prod) and inflates sigma.
        # A test measured sigma_deb=1.15 for a true value near 0.19, a 6x error.
        # LCB then stayed negative and core fell to min_core_size.
        #
        # Subtract the remaining initial-value contribution first:
        #     sigma_hat = (sigma_bar - prod*init) / (1 - prod)
        # ---------------------------------------------------------------
        val = (float(self.sigma_bar[j]) - prod * sigma_init) / denom

        return max(val, self.sigma_floor)

    def _lcb_score(self, j: int) -> float:
        """
        [B1] Lower confidence bound replacing v1's saturating formula.

            score = |mu_bar| - kappa * sigma_bar

        High uncertainty is subtracted rather than used as a divisor, so it
        cannot cause saturation. This follows bandit LCB/UCB practice and
        CASEC's use of payoff variance for edge sparsification.
        """
        j = int(j)
        mu = self.debiased_mu(j)
        sigma = self._safe_sigma(self.debiased_sigma(j))

        return float(max(mu, 0.0) - self.kappa * sigma)

    def _priority(self, j: int) -> float:
        """Priority for fill/prune on the same scale as core selection."""
        if self.core_rule == "p_core":
            j = int(j)
            mu = max(float(self.debiased_mu(j)), 0.0)
            sigma = self._safe_sigma(self.sigma_bar.get(j, 1.0))
            return float(mu / (sigma + self.eps))

        return self._lcb_score(j)

    def _rank(self, candidates=None) -> List[int]:
        if candidates is None:
            candidates = self.neighbor_ids

        cands = [int(j) for j in candidates if int(j) in self.mu_bar]

        return sorted(cands, key=lambda j: self._priority(j), reverse=True)

    # =====================================================================
    # [B2] Adaptive k.
    # =====================================================================

    def _effective_max_k(self) -> int:
        """
        If enabled, scale target k with normalized influence entropy.

        Concentration on one or two neighbours gives low entropy and small k;
        diffuse influence gives high entropy and larger k because no neighbour
        clearly dominates.

        This makes adaptive capacity an actual mechanism rather than a label.
        """
        if not self.adaptive_k:
            return int(self.max_core_size)

        vals = np.array(
            [max(float(self.debiased_mu(j)), 0.0) for j in self.neighbor_ids],
            dtype=np.float64,
        )  # [n_neighbors]

        total = float(np.sum(vals))

        if total <= self.eps or len(vals) <= 1:
            # No positive structural-capacity evidence must not allocate the
            # largest explicit core.  The lower bound remains a safety valve.
            return int(self.min_core_size)

        p = vals / total                                  # [n_neighbors]
        p = np.clip(p, 1e-12, 1.0)

        entropy = float(-np.sum(p * np.log(p)))
        max_entropy = float(np.log(len(vals)))

        # frac in [0,1]: zero is concentrated and one fully diffuse.
        frac = entropy / max(max_entropy, 1e-12)

        k = int(np.ceil(
            self.adaptive_k_min
            + frac * (self.max_core_size - self.adaptive_k_min)
        ))

        return int(np.clip(k, self.adaptive_k_min, self.max_core_size))

    def _apply_capacity(self, candidate_core: Set[int]) -> Set[int]:
        core = set(int(j) for j in candidate_core if int(j) in self.mu_bar)

        eff_max = self._effective_max_k()

        self.n_core_updates += 1

        if len(core) < self.min_core_size:
            self.n_hit_min += 1

            for j in self._rank(self.neighbor_ids):
                core.add(int(j))
                if len(core) >= self.min_core_size:
                    break

        if len(core) > eff_max:
            self.n_hit_max += 1
            core = set(self._rank(core)[:eff_max])

        return core

    def _record_core_change(self, old_core, new_core):
        old_core = set(old_core)
        new_core = set(new_core)

        self.last_promoted = set(new_core - old_core)
        self.last_demoted = set(old_core - new_core)
        self.last_core_switch_count = len(old_core ^ new_core)

        self.core_history.append(set(new_core))

        if len(self.core_history) > 500:
            self.core_history = self.core_history[-500:]

    # =====================================================================
    # Stage 0 seeding
    # =====================================================================

    def initialize_from_weak_prior(self, prior_scores: Dict[int, float]):
        """Preserve v1 behaviour."""
        ranked = sorted(
            [(int(j), float(prior_scores.get(j, 0.0))) for j in self.neighbor_ids],
            key=lambda x: x[1],
            reverse=True,
        )

        seed_k = min(
            max(0, int(self.weak_prior_top_k)),
            self.max_core_size,
            len(self.neighbor_ids),
        )

        chosen = [j for j, _ in ranked[:seed_k]]

        old_core = set(self.core_set)

        self.seeded_core_set = set(chosen)
        self.core_set = set(chosen)
        self.prev_core_set = set(chosen)

        for j in self.neighbor_ids:
            self.p_core[j] = 0.58 if j in self.seeded_core_set else 0.42

        self._record_core_change(old_core, self.core_set)

    def set_fixed_core(self, core_ids):
        old_core = set(self.core_set)

        cand = set(
            int(j) for j in core_ids if int(j) in self.neighbor_ids
        )
        cand = self._apply_capacity(cand)

        self.core_set = set(cand)
        self.prev_core_set = set(cand)
        self.seeded_core_set = set(cand)

        for j in self.neighbor_ids:
            self.p_core[j] = 0.58 if j in self.core_set else 0.42

        self._record_core_change(old_core, self.core_set)

    # =====================================================================
    # Belief updates.
    # =====================================================================

    def _debiased_arr(self, idx: np.ndarray):
        """
        Vectorized debiased_mu/debiased_sigma over index set idx. The formulas
        remain identical to the scalar reference methods; only execution
        changes from per-j calls to arrays.
        """
        prod = self._bias_corr_arr[idx]
        denom = 1.0 - prod
        mu_init = self._mu_init_arr[idx]
        sigma_init = self._sigma_init_arr[idx]

        small = denom < 1e-3
        denom_safe = np.where(small, 1.0, denom)

        mu_deb = np.where(small, mu_init, (self._mu_arr[idx] - prod * mu_init) / denom_safe)
        sig_deb = np.where(
            small,
            np.maximum(sigma_init, self.sigma_floor),
            np.maximum((self._sigma_arr[idx] - prod * sigma_init) / denom_safe, self.sigma_floor),
        )
        return mu_deb, sig_deb

    def update_pair(self, j: int, mu: float, sigma: float):
        """Update one pair through the single-item vectorized batch path."""
        self.update_batch({int(j): (mu, sigma)})

    def update_batch(
        self, mu_sigma_dict, *, select_core: bool = True
    ) -> Tuple[Set[int], Set[int]]:
        """
        Preserve the v1 signature: accept {j:(mu,sigma)} and return promoted/demoted.

        [GPU contract section 1.2] Run Eq. 11-13 EWMA and Eq. 14 bias
        correction for all j in one NumPy operation instead of an
        O(n_neighbors) Python loop at every slow-timescale ego update. Each
        element uses the exact former update_pair formula.
        """
        js, mus, sigmas = [], [], []
        for j, pair_value in mu_sigma_dict.items():
            if pair_value is None:
                continue
            jj = int(j)
            if jj not in self._pos:
                continue
            mu, sigma = pair_value
            js.append(jj)
            mus.append(float(mu))
            sigmas.append(self._safe_sigma(sigma))

        if len(js) > 0:
            idx = np.array([self._pos[j] for j in js], dtype=np.int64)
            mus_arr = np.asarray(mus, dtype=np.float64)
            sigmas_arr = np.asarray(sigmas, dtype=np.float64)

            self._n_updates_arr[idx] += 1
            t = self._n_updates_arr[idx].astype(np.float64)

            # [B3] Robbins-Monro schedule; formula unchanged.
            decay_factor = (
                t ** self.alpha_decay if self.alpha_decay > 0.0 else np.ones_like(t)
            )
            # Bounded uncertainty is required for the within-regime
            # Robbins--Monro argument.  The belief still stores the raw
            # uncertainty; only its learning-rate modulation is clipped.
            sigma_for_alpha = np.clip(
                sigmas_arr, self.sigma_floor, self.sigma_alpha_max
            )
            alpha = self.lambda_0 / (
                decay_factor * (1.0 + self.uncertainty_scale * sigma_for_alpha)
            )
            alpha = np.clip(alpha, 0.0, 1.0)

            # [B4] Preserve mu sign.
            self._mu_arr[idx] = (1.0 - alpha) * self._mu_arr[idx] + alpha * mus_arr
            self._sigma_arr[idx] = (
                (1.0 - alpha) * self._sigma_arr[idx] + alpha * sigmas_arr
            )

            # Adam-style bias correction; formula unchanged.
            self._bias_corr_arr[idx] = self._bias_corr_arr[idx] * (1.0 - alpha)

            # p_core is a probability-like visualization of the structural
            # priority score. It is not statistically calibrated and is not a
            # separate control signal.
            mu_deb, sig_deb = self._debiased_arr(idx)
            effective_sigma = np.maximum(sig_deb, self.sigma_floor)

            lcb = np.maximum(mu_deb, 0.0) - self.kappa * effective_sigma
            normalized_lcb = (lcb - self.tau) / max(self.tau, 1e-4)
            p = 1.0 / (1.0 + np.exp(-np.clip(normalized_lcb, -10.0, 10.0)))

            # Always record debug state; hasattr previously skipped the first write.
            self._last_lcb_debug = {
                "mu_deb_mean": float(np.mean(np.abs(mu_deb))),
                "penalty_mean": float(np.mean(self.kappa * effective_sigma)),
                "lcb_mean": float(np.mean(lcb)),
                "p_mean": float(np.mean(p)),
            }

            # Vectorized mu_deb/sig_deb/p arrays must be assigned directly.
            # Index them only when idx is known to match the current context.
            if isinstance(idx, np.ndarray) and idx.dtype == bool:
                # Boolean-mask index.
                self._p_core_arr[idx] = p[idx] if p.shape == idx.shape else p
            else:
                # Integer index array.
                np.put(self._p_core_arr, idx, p)

            # Synchronize dictionary views and history; all computation already
            # occurred in the NumPy block above.
            for pos, jj in zip(idx.tolist(), js):
                mu_v = float(self._mu_arr[pos])
                sig_v = float(self._sigma_arr[pos])
                p_v = float(self._p_core_arr[pos])

                self.mu_bar[jj] = mu_v
                self.sigma_bar[jj] = sig_v
                self.n_updates[jj] = int(self._n_updates_arr[pos])
                self._bias_corr[jj] = float(self._bias_corr_arr[pos])
                self.p_core[jj] = p_v

                self.mu_history[jj].append(mu_v)
                self.sigma_history[jj].append(sig_v)
                self.p_history[jj].append(p_v)
                if len(self.mu_history[jj]) > 500:
                    del self.mu_history[jj][:-500]
                if len(self.sigma_history[jj]) > 500:
                    del self.sigma_history[jj][:-500]
                if len(self.p_history[jj]) > 500:
                    del self.p_history[jj][:-500]

        if not bool(select_core):
            return set(), set()
        return self._update_core_set()

    def update_evidence(self, mu_sigma_dict) -> None:
        """Update C evidence without executing an allocation rule."""
        self.update_batch(mu_sigma_dict, select_core=False)

    # =====================================================================
    # Core selection.
    # =====================================================================

    def _select_core_lcb(self) -> Set[int]:
        """
        Default v2 rule.

        Enter core when lcb_score>tau; remain when lcb_score>tau_hold, where
        tau_hold<tau provides hysteresis.

        Hysteresis preserves v1's anti-chatter intent on the LCB scale instead
        of the saturated p_core scale.
        """
        # [GPU contract section 1.4] The dual-threshold rule is elementwise
        # Boolean logic, vectorized across neighbours in one NumPy operation.
        # The formula is unchanged.
        ratio = (
            self.tau_out / self.tau_in if self.tau_in > 1e-8 else 0.75
        )
        tau_hold = self.tau * float(np.clip(ratio, 0.0, 1.0))

        idx_all = np.arange(len(self.neighbor_ids), dtype=np.int64)
        mu_deb, sig_deb = self._debiased_arr(idx_all)
        g = np.maximum(mu_deb, 0.0) - self.kappa * np.maximum(sig_deb, self.sigma_floor)  # [n]

        was_in_prev = np.array(
            [j in self.prev_core_set for j in self.neighbor_ids], dtype=bool
        )

        enter = g > self.tau
        stay = was_in_prev & (g > tau_hold)
        keep = enter | stay

        new_core = set(int(j) for j, k in zip(self.neighbor_ids, keep.tolist()) if k)

        return new_core

    def _select_core_signed(self) -> Set[int]:
        """
        Signed rule balancing helpful and harmful agents.

        MAGIC (2026) shows strong influence need not be beneficial. Ranking
        only by |mu| can fill the core with one sign. Allocate signed_balance
        of slots to harmful agents, often important to avoid, and the rest to
        helpful agents.

        This variant uses sign to balance core capacity, unlike MAGIC's use of
        sign for reward filtering.
        """
        eff_max = self._effective_max_k()

        n_harm = int(round(self.signed_balance * eff_max))
        n_help = eff_max - n_harm

        harmful = [
            j for j in self.neighbor_ids
            if float(self.mu_bar[j]) < 0.0 and self._lcb_score(j) > self.tau
        ]
        helpful = [
            j for j in self.neighbor_ids
            if float(self.mu_bar[j]) >= 0.0 and self._lcb_score(j) > self.tau
        ]

        harmful.sort(key=lambda j: self._lcb_score(j), reverse=True)
        helpful.sort(key=lambda j: self._lcb_score(j), reverse=True)

        new_core = set(harmful[:n_harm]) | set(helpful[:n_help])

        # Fill shortages from the other sign without wasting capacity.
        if len(new_core) < eff_max:
            leftovers = [
                j for j in self._rank(self.neighbor_ids)
                if j not in new_core and self._lcb_score(j) > self.tau
            ]
            for j in leftovers:
                new_core.add(j)
                if len(new_core) >= eff_max:
                    break

        return new_core

    def _select_core_p(self) -> Set[int]:
        """Legacy v1 rule retained for before-correction ablation."""
        new_core = set()

        for j in self.neighbor_ids:
            p = float(self.p_core[j])

            if p > self.tau_in:
                new_core.add(j)
            elif (j in self.prev_core_set) and (p >= self.tau_out):
                new_core.add(j)

        return new_core

    def select_core_from_external_scores(
        self,
        scores: Dict[int, float],
        target_size: Optional[int] = None,
    ) -> Tuple[Set[int], Set[int]]:
        """Apply an explicit experimental allocation rule with a fixed budget.

        The structural belief remains updated from C regardless of this method.
        This hook is only for controlled allocation ablations such as C-core
        versus ``|D|``-core at the same per-ego budget; it does not turn D into
        a structural belief signal.
        """
        old_core = set(self.core_set)
        self.prev_core_set = set(old_core)
        effective_max = self._effective_max_k()
        requested = len(old_core) if target_size is None else int(target_size)
        k = int(np.clip(requested, self.min_core_size, effective_max))
        ranked = sorted(
            self.neighbor_ids,
            key=lambda j: float(scores.get(int(j), float("-inf"))),
            reverse=True,
        )
        self.n_core_updates += 1
        if k == self.min_core_size:
            self.n_hit_min += 1
        if k == effective_max:
            self.n_hit_max += 1
        self.core_set = set(int(j) for j in ranked[:k])
        self._record_core_change(old_core, self.core_set)
        return set(self.last_promoted), set(self.last_demoted)

    def _update_core_set(self) -> Tuple[Set[int], Set[int]]:
        old_core = set(self.core_set)
        self.prev_core_set = set(self.core_set)

        if self.core_rule == "lcb":
            new_core = self._select_core_lcb()
        elif self.core_rule == "signed":
            new_core = self._select_core_signed()
        else:
            new_core = self._select_core_p()

        new_core = self._apply_capacity(new_core)

        self.core_set = set(new_core)
        self._record_core_change(old_core, self.core_set)

        return set(self.last_promoted), set(self.last_demoted)

    # =====================================================================
    # Accessors preserving the v1 API.
    # =====================================================================

    def get_core_set(self) -> Set[int]:
        return set(self.core_set)

    def get_peripheral_set(self) -> Set[int]:
        return set(self.neighbor_ids) - set(self.core_set)

    def get_state_for_neighbor(self, j: int) -> Dict[str, float]:
        j = int(j)

        return {
            # Publish debiased values because peripheral semantic assignment
            # requires a correctly scaled estimate.
            "mu_bar": float(self.debiased_mu(j)),      # Signed.
            "sigma_bar": float(self.debiased_sigma(j)),
            "capacity_bar": float(max(self.debiased_mu(j), 0.0)),
            "sigma_capacity_bar": float(self.debiased_sigma(j)),
            "mu_bar_raw": float(self.mu_bar[j]),
            "p_core": float(self.p_core[j]),
            "in_core": float(j in self.core_set),
            "in_seed_core": float(j in self.seeded_core_set),
            # Direct peripheral-memory fields.
            "lcb_score": float(self._lcb_score(j)),
            "n_updates": int(self.n_updates[j]),
        }

    def get_state_dict(self) -> Dict[int, Dict[str, float]]:
        return {int(j): self.get_state_for_neighbor(j) for j in self.neighbor_ids}

    def get_mean_uncertainty(self) -> float:
        vals = [float(self.sigma_bar[j]) for j in self.neighbor_ids]
        return float(np.mean(vals)) if vals else 0.0

    def get_temporal_variance(self, window: int = 50) -> float:
        """
        Paper note: v1 reported approximately 1e-8 as stability. That value is
        effectively zero movement and indicates freezing rather than stability.
        Report it with normalised_temporal_variance below.
        """
        vals = []

        for j in self.neighbor_ids:
            hist = self.mu_history[j][-int(window):]
            if len(hist) > 1:
                vals.append(float(np.var(hist)))

        return float(np.mean(vals)) if vals else 0.0

    def get_normalised_temporal_variance(self, window: int = 50) -> float:
        """
        Temporal variance normalized by mu's own scale.

            nvar = var(mu_t) / (mean(|mu_t|)^2 + eps)

        The dimensionless result is comparable across methods and does not
        appear artificially small merely because mu is small.
        """
        vals = []

        for j in self.neighbor_ids:
            hist = np.asarray(self.mu_history[j][-int(window):], dtype=np.float64)

            if hist.shape[0] > 1:
                scale = float(np.mean(np.abs(hist))) ** 2
                vals.append(float(np.var(hist)) / (scale + 1e-12))

        return float(np.mean(vals)) if vals else 0.0

    def get_core_switch_count(self) -> int:
        return int(self.last_core_switch_count)

    def reset_switch_counter(self):
        self.last_promoted = set()
        self.last_demoted = set()
        self.last_core_switch_count = 0
        self.prev_core_set = set(self.core_set)

    def get_last_promoted(self) -> Set[int]:
        return set(self.last_promoted)

    def get_last_demoted(self) -> Set[int]:
        return set(self.last_demoted)

    def get_mean_abs_mu(self) -> float:
        vals = [max(float(self.debiased_mu(j)), 0.0) for j in self.neighbor_ids]
        return float(np.mean(vals)) if vals else 0.0

    def get_max_p_core(self) -> float:
        vals = [float(self.p_core[j]) for j in self.neighbor_ids]
        return float(np.max(vals)) if vals else 0.0

    # =====================================================================
    # Uncertainty inflation: controlled forgetting after structural shifts.
    # =====================================================================

    def inflate_uncertainty(
        self,
        factor: float = 2.5,
        t_reset: int = 1,
        pairs: Optional[List[int]] = None,
        sigma_ceiling: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Call when residual or matrix triggers indicate a structural shift.

        PROBLEM ADDRESSED
        -----------------
        The Robbins-Monro schedule alpha~1/t^d is required for the convergence
        theorem, but it creates a paradox: the longer the system learns, the
        smaller alpha becomes and the more resistant the system becomes to new
        evidence. After a few hundred steps, new evidence can barely move the
        belief. In a stationary environment this is a feature. In the
        nonstationary environment studied here it is a defect: the structure
        can genuinely change while the system refuses to revise its belief.

        An everyday analogy is an impression of a person formed over ten
        years. If that person changes job, circumstances, and behaviour, an
        update still weighted by the full ten-year history could take years to
        recognize the change. A rational observer instead acknowledges that
        the context changed, lowers confidence in the old impression, and
        reopens the learning rate. This method performs the same operation.

        MECHANISM — BOTH STEPS ARE REQUIRED
        -----------------------------------
        1. RE-ANCHOR: use the current debiased estimate as the new anchor. This
           differs from a reset to zero: learned information is preserved and
           only confidence in that information is reduced.
        2. INFLATE AND REOPEN LEARNING: multiply sigma by `factor` and reset the
           counter t to `t_reset`, returning alpha to a high value.

        CASCADING EFFECT
        ----------------
        Increased sigma lowers LCB=|mu|-kappa*sigma, so fewer neighbours exceed
        tau and the CORE CONTRACTS AUTOMATICALLY. The system becomes cautious
        while it does not understand the new situation, then expands the core
        again after relearning. No additional manual rule is required.

        This is covariance inflation from Kalman filtering and is related to
        discounted/sliding-window UCB for nonstationary bandits.

        Args:
            factor: sigma multiplier; 2.0-3.0 is reasonable.
            t_reset: new counter value; 1 fully reopens learning.
            pairs: affected pairs, or None for all.
            sigma_ceiling: post-inflation cap, or SIGMA_INIT when None.

        Returns:
            Statistics dictionary for logging.
        """
        ids = self.neighbor_ids if pairs is None else [
            int(j) for j in pairs if int(j) in self.mu_bar
        ]

        ceiling = (
            float(self.SIGMA_INIT) if sigma_ceiling is None
            else float(sigma_ceiling)
        )

        before = float(np.mean(
            [self.debiased_sigma(j) for j in ids]
        )) if ids else 0.0

        for j in ids:
            # Step 1: re-anchor at the best current estimate.
            mu_hat = self.debiased_mu(j)
            sig_hat = self.debiased_sigma(j)

            # Step 2: inflate under a ceiling.
            new_sigma = min(
                max(sig_hat * float(factor), self.sigma_floor),
                ceiling,
            )

            self.mu_bar[j] = mu_hat
            self.sigma_bar[j] = new_sigma

            # With _bias_corr=1.0, the newly set state is the anchor returned by
            # debiased_* until sufficient new evidence accumulates.
            self._mu_init[j] = mu_hat
            self._sigma_init[j] = new_sigma
            self._bias_corr[j] = 1.0

            # Reopen the learning rate.
            self.n_updates[j] = int(max(0, t_reset))

            # ---------------------------------------------------------
            # Synchronize authoritative arrays. update_batch and core selection
            # no longer read dictionaries; omission would leave vectorized paths
            # on pre-inflation state, the synchronization-order failure class
            # described in GPU contract section 1.3.
            # ---------------------------------------------------------
            pos = self._pos[j]
            self._mu_arr[pos] = mu_hat
            self._sigma_arr[pos] = new_sigma
            self._mu_init_arr[pos] = mu_hat
            self._sigma_init_arr[pos] = new_sigma
            self._bias_corr_arr[pos] = 1.0
            self._n_updates_arr[pos] = int(max(0, t_reset))

        self.n_inflations += 1

        after = float(np.mean(
            [self.debiased_sigma(j) for j in ids]
        )) if ids else 0.0

        return {
            "n_pairs_inflated": int(len(ids)),
            "sigma_before": before,
            "sigma_after": after,
            "factor": float(factor),
            "n_inflations_total": int(self.n_inflations),
        }

    # =====================================================================
    # New diagnostics.
    # =====================================================================

    def get_saturation_stats(self) -> Dict[str, float]:
        """
        [B2] Required paper statistic.

        If hit_max_rate is near one, core size is a hard ceiling rather than a
        learned outcome, so adaptive-capacity claims are unsupported.
        """
        n = max(1, self.n_core_updates)

        return {
            "n_core_updates": int(self.n_core_updates),
            "hit_max_rate": float(self.n_hit_max) / float(n),
            "hit_min_rate": float(self.n_hit_min) / float(n),
            "current_core_size": int(len(self.core_set)),
            "effective_max_k": int(self._effective_max_k()),
            "hard_max_core_size": int(self.max_core_size),
        }

    def get_signed_stats(self) -> Dict[str, float]:
        """[B4] Signed statistics used by semantic slots."""
        mus = np.array(
            [float(self.mu_bar[j]) for j in self.neighbor_ids], dtype=np.float64
        )

        if mus.size == 0:
            return {
                "n_helpful": 0, "n_harmful": 0, "n_neutral": 0,
                "mean_signed_mu": 0.0, "helpful_harmful_ratio": 0.0,
            }

        helpful = int(np.sum(mus > self.tau))
        harmful = int(np.sum(mus < -self.tau))
        neutral = int(mus.size - helpful - harmful)

        return {
            "n_helpful": helpful,
            "n_harmful": harmful,
            "n_neutral": neutral,
            "mean_signed_mu": float(np.mean(mus)),
            "helpful_harmful_ratio": float(helpful) / float(max(1, harmful)),
        }

    def get_population_debug_stats(self) -> Dict:
        mu_abs = [abs(float(self.mu_bar[j])) for j in self.neighbor_ids]
        sig = [float(self.sigma_bar[j]) for j in self.neighbor_ids]
        lcb = [self._lcb_score(j) for j in self.neighbor_ids]

        out = {
            "ego_id": int(self.ego_id),
            "n_neighbors": int(len(self.neighbor_ids)),
            "core_size": int(len(self.core_set)),
            "peripheral_size": int(len(self.get_peripheral_set())),
            "core_rule": self.core_rule,
            "kappa": float(self.kappa),
            "tau": float(self.tau),
            "mean_abs_mu": float(np.mean(mu_abs)) if mu_abs else 0.0,
            "max_abs_mu": float(np.max(mu_abs)) if mu_abs else 0.0,
            "mean_sigma": float(np.mean(sig)) if sig else 0.0,
            "min_sigma": float(np.min(sig)) if sig else 0.0,
            "max_sigma": float(np.max(sig)) if sig else 0.0,
            "mean_lcb": float(np.mean(lcb)) if lcb else 0.0,
            "max_lcb": float(np.max(lcb)) if lcb else 0.0,
            "n_above_tau": int(sum(1 for s in lcb if s > self.tau)),
            "last_core_switch_count": int(self.last_core_switch_count),
            "core_set": sorted(int(x) for x in self.core_set),
        }

        out.update(self.get_saturation_stats())
        out.update(self.get_signed_stats())

        return out


# Compatibility alias.
BayesLightBeliefState = BayesLightBeliefState
