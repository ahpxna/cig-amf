"""Run Paper-B selector-isolation and end-to-end allocation panels.

The central test is deliberately narrow: ``C`` and ``|D|`` select the same
number of explicit neighbours under the same seed, architecture, and training
budget.  C is the structural-capacity selector; |D| is the behavioural
direction confounder.  This runner does not treat the result as a causal
identification test—that remains Paper A/H1.
"""

import argparse
import copy
import csv
import json
import os
import tempfile

import numpy as np

try:
    from exp_common import ROOT, ensure_dir
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir

import run_experiment as RE
from envs.causal_adapter import resolve_env_adapter
try:
    from run_h2_selectivity import (
        _capture_frozen_learning_checkpoint,
        _restore_frozen_learning_checkpoint,
    )
except ModuleNotFoundError:
    from scripts.run_h2_selectivity import (
        _capture_frozen_learning_checkpoint,
        _restore_frozen_learning_checkpoint,
    )


VARIANTS = {
    "C-Core": "structural_capacity",
    "AbsD-Core": "behavioral_direction",
    "Random-Core": "random",
    "Correlation-Core": "observational_correlation",
    "Attention-Core": "attention",
    "WeakPrior-Core": "weak_prior",
    "Oracle-C-Core": "oracle_capacity",
    "Full-Explicit": "full_explicit",
}


def _decision_probe(runner, n_states, seed):
    """Evaluate policy/value outputs on a held-out common state bank."""
    bank = runner.env.sample_state_bank(
        n_states=int(n_states), burn_in=3, bank_seed=int(seed)
    )
    outer = runner.env.clone_state()
    logits, values, actions = [], [], []
    try:
        for state in bank:
            runner.env.restore_state(copy.deepcopy(state))
            obs = runner.env._get_obs_all()
            selected, cache = runner._select_actions_population(
                obs, apply_forcing=False
            )
            logits.append(np.asarray(cache["policy_logits"], dtype=np.float64))
            values.append(np.asarray(
                [cache["value_cache"][ego] for ego in range(runner.n_agents)],
                dtype=np.float64,
            ))
            # Fidelity compares the policy decision rule, not epsilon-forced
            # sampled actions.  The cached logits are already valid-action
            # masked by PolicyValueNet.
            actions.append(np.argmax(
                np.asarray(cache["policy_logits"], dtype=np.float64), axis=-1
            ).astype(np.int64))
    finally:
        runner.env.restore_state(outer)
    return {
        "logits": np.stack(logits, axis=0),
        "values": np.stack(values, axis=0),
        "actions": np.stack(actions, axis=0),
    }


def _decision_fidelity(probe, reference):
    return {
        "policy_logit_l2_to_full_explicit": float(np.mean(
            (probe["logits"] - reference["logits"]) ** 2
        ) ** 0.5),
        "value_mae_to_full_explicit": float(np.mean(np.abs(
            probe["values"] - reference["values"]
        ))),
        "action_agreement_to_full_explicit": float(np.mean(
            probe["actions"] == reference["actions"]
        )),
    }


def _atomic_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp-paper-b-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _mean(values):
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _cfg(seed, core_budget, selector="structural_capacity"):
    cfg = RE.default_cfg()
    cfg.update({
        "seed": int(seed),
        "k0_warmup": 0,
        "core_selection_mode": str(selector),
        "belief_adaptive_k": False,
        "min_core_size": int(core_budget),
        "belief_adaptive_k_min": int(core_budget),
        "max_core_size": int(core_budget),
        "periph_require_full_signature": True,
        "periph_allow_legacy_items": False,
        "strict_causal_profile": True,
        "semantic_router_frozen": True,
        # Allocation truth is standardized at h=1.  This removes the
        # continuation-policy mismatch that otherwise makes C^rho from an
        # environment-scripted oracle incomparable with a learned-policy
        # response estimator.  Paper-A latency remains an upstream diagnostic;
        # Paper-B allocation does not require a long-horizon oracle ranking.
        "causal_horizon": 1,
        "proxy_n_horizons": 1,
    })
    return cfg


def _make_env(seed, phase_length=40, n_agents=24):
    return RE.make_main_env(
        task_mode="behavioral_drift",
        n_agents=int(n_agents),
        max_steps=30,
        phase_length=int(phase_length),
        seed=int(seed),
    )


def _apply_variant_allocation_state(runner, selector, core_budget):
    """Reapply selector budget after checkpoint runtime restoration."""
    if selector == "full_explicit":
        for belief in runner.belief_modules.values():
            size = len(belief.neighbor_ids)
            belief.adaptive_k = False
            belief.min_core_size = size
            belief.max_core_size = size
            belief.adaptive_k_min = size
            belief.set_fixed_core(belief.neighbor_ids)
        runner.pair_rel_module.reconcile_core_sets(
            {ego: belief.get_core_set() for ego, belief in runner.belief_modules.items()}
        )
        return
    for belief in runner.belief_modules.values():
        belief.adaptive_k = False
        belief.min_core_size = int(core_budget)
        belief.max_core_size = int(core_budget)
        belief.adaptive_k_min = int(core_budget)


def _prepare_variant_runner(
    variant, seed, core_budget, device, checkpoint, n_agents=24
):
    selector = VARIANTS[variant]
    cfg = _cfg(seed, core_budget, selector=selector)
    env = _make_env(seed, n_agents=n_agents)
    runner = RE.make_runner("Final-CIGAMF", env, cfg, device)
    _restore_frozen_learning_checkpoint(runner, checkpoint)
    _apply_variant_allocation_state(runner, selector, core_budget)
    return runner


def _top_k(scores, ego, core_budget):
    candidates = [int(j) for j in scores if int(j) != int(ego)]
    return set(sorted(
        candidates, key=lambda j: float(scores[j]), reverse=True
    )[:int(core_budget)])


def _policy_context_for_state(runner, state):
    """Return a side-effect-free frozen current-policy context for one state.

    The selector-isolation bank must not let each variant choose a different
    factual co-action or target policy.  We therefore query one neutral runner
    once per state, keep deterministic argmax co-actions plus the pre-forcing
    policy probabilities, and restore RNG/forcer/environment state afterwards.
    """
    outer = runner.env.clone_state()
    np_state = np.random.get_state()
    forcer = getattr(runner, "forcer", None)
    forcer_state = (
        copy.deepcopy(forcer.state_dict())
        if forcer is not None and callable(getattr(forcer, "state_dict", None))
        else None
    )
    try:
        runner.env.restore_state(copy.deepcopy(state))
        obs_all = runner.env._get_obs_all()
        _sampled, cache = runner._select_actions_population(
            obs_all, apply_forcing=False
        )
        logits = np.asarray(cache["policy_logits"], dtype=np.float64)
        actions = np.argmax(logits, axis=-1).astype(np.int64)
        probs = np.asarray(cache["policy_probs"], dtype=np.float64)
        valid = np.asarray(cache["valid_action_masks"], dtype=bool)
        if probs.shape != valid.shape or logits.shape != probs.shape:
            raise RuntimeError("frozen policy context has inconsistent tensor shapes")
        return {
            "actions": actions,
            "policy_probs": probs,
            "valid_action_masks": valid,
            "policy_logits": logits,
        }
    finally:
        runner.env.restore_state(outer)
        np.random.set_state(np_state)
        if forcer_state is not None:
            forcer.load_state_dict(forcer_state)


def _oracle_capacity_direction_for_state(
    env, state, horizon, discount, trials, seed, target_policy_probs=None
):
    """Compute one-step clone-state oracle C* and current-policy D*.

    ``target_policy_probs`` must come from the frozen common policy at this
    exact state.  H=1 is required so the oracle and learned allocation score do
    not silently differ only because their continuation regimes differ.
    """
    if int(horizon) != 1:
        raise ValueError("Paper-B selector oracle is standardized to horizon=1")
    if target_policy_probs is None:
        raise ValueError("current frozen target-policy probabilities are required")
    target_policy_probs = np.asarray(target_policy_probs, dtype=np.float64)
    expected = (int(env.n_agents), int(env.get_action_dim()))
    if target_policy_probs.shape != expected:
        raise ValueError(
            f"target_policy_probs must have shape {expected}, got {target_policy_probs.shape}"
        )
    outer = env.clone_state()
    capacities, directions = {}, {}
    try:
        for ego in range(int(env.n_agents)):
            c_row, d_row = {}, {}
            for source in range(int(env.n_agents)):
                if source == ego:
                    continue
                pair_seed = (
                    (int(seed) + 1) * 1000003
                    + int(ego) * 1009 + int(source) * 9176
                )
                valid_actions = np.flatnonzero(
                    resolve_env_adapter(env).valid_action_mask(source)
                )
                if valid_actions.size == 0:
                    raise RuntimeError(
                        "oracle allocation bank encountered an empty valid-action set"
                    )
                # With exactly one available action the max--min capacity and
                # policy-versus-reference directional contrast are both
                # mathematically zero.  This is a valid null causal pair, not
                # an oracle failure, and matters for masked external actions.
                if valid_actions.size == 1:
                    c_row[source] = 0.0
                    d_row[source] = 0.0
                    continue
                env.restore_state(copy.deepcopy(state))
                env.set_behaviour_override("cooperative")
                pi_full = np.asarray(
                    target_policy_probs[int(source)], dtype=np.float64
                )
                if (
                    not np.all(np.isfinite(pi_full))
                    or np.any(pi_full < 0.0)
                    or float(pi_full[valid_actions].sum()) <= 0.0
                    or float(pi_full[~np.isin(np.arange(pi_full.size), valid_actions)].sum()) > 1e-8
                ):
                    raise ValueError(
                        "frozen target policy violates the valid-action probability contract"
                    )
                pi = pi_full[valid_actions]
                pi = pi / float(pi.sum())
                q = np.full(valid_actions.size, 1.0 / valid_actions.size)
                action_values = []
                for action in valid_actions:
                    env.restore_state(copy.deepcopy(state))
                    env.set_behaviour_override("cooperative")
                    response = env.compute_oracle_lag_response_from_current_state(
                        ego_id=ego,
                        agent_j=source,
                        intervention_action=int(action),
                        horizon=int(horizon),
                        n_trials=int(trials),
                        forced_step=0,
                        continuation_policy=env.scripted_policy,
                        crn_seed=pair_seed,
                        discount=float(discount),
                    )
                    action_values.append(float(response["discounted_response"]))
                values = np.asarray(action_values, dtype=np.float64)
                c_row[source] = float(np.max(values) - np.min(values))
                d_row[source] = float(np.dot(pi - q, values))
            capacities[ego] = c_row
            directions[ego] = d_row
    finally:
        env.restore_state(outer)
    return capacities, directions


def _oracle_capacity_for_state(
    env, state, horizon, discount, trials, seed, target_policy_probs=None
):
    if target_policy_probs is None:
        # C itself is policy-contrast free at h=1.  Construct a valid reference
        # only so the shared C/D oracle helper can execute; callers that consume
        # D must pass the actual frozen current policy explicitly.
        action_dim = int(env.get_action_dim())
        rows = []
        adapter = resolve_env_adapter(env)
        for agent in range(int(env.n_agents)):
            mask = np.asarray(adapter.valid_action_mask(agent), dtype=bool)
            row = np.zeros(action_dim, dtype=np.float64)
            row[mask] = 1.0 / float(np.count_nonzero(mask))
            rows.append(row)
        target_policy_probs = np.stack(rows, axis=0)
    capacities, _ = _oracle_capacity_direction_for_state(
        env, state, horizon, discount, trials, seed,
        target_policy_probs=target_policy_probs,
    )
    return capacities




def _mean_score_tables(tables):
    """Elementwise mean of oracle ego->neighbour score tables."""
    if not tables:
        raise ValueError("oracle score table list is empty")
    egos = set(tables[0])
    if any(set(table) != egos for table in tables[1:]):
        raise ValueError("oracle replicas disagree on ego keys")
    out = {}
    for ego in sorted(egos):
        neighbours = set(tables[0][ego])
        if any(set(table[ego]) != neighbours for table in tables[1:]):
            raise ValueError("oracle replicas disagree on neighbour keys")
        out[int(ego)] = {
            int(j): float(np.mean([float(table[ego][j]) for table in tables]))
            for j in neighbours
        }
    return out


def _topk_jaccard(left_scores, right_scores, ego, k, *, absolute=False):
    def transform(row):
        return {
            int(j): abs(float(v)) if absolute else float(v)
            for j, v in row.items()
        }
    left = _top_k(transform(left_scores[int(ego)]), ego, k)
    right = _top_k(transform(right_scores[int(ego)]), ego, k)
    union = left | right
    return 1.0 if not union else float(len(left & right) / len(union))


def _oracle_replica_summary(replicas, core_budget):
    """Return averaged C/D plus worst-ego top-k repeatability across replicas."""
    if len(replicas) < 2:
        raise ValueError("oracle ranking stability requires at least two replicas")
    capacities = [item[0] for item in replicas]
    directions = [item[1] for item in replicas]
    c_mean = _mean_score_tables(capacities)
    d_mean = _mean_score_tables(directions)
    c_jaccards, d_jaccards = [], []
    base_c, base_d = capacities[0], directions[0]
    for replica_c, replica_d in zip(capacities[1:], directions[1:]):
        for ego in base_c:
            c_jaccards.append(
                _topk_jaccard(base_c, replica_c, ego, core_budget, absolute=False)
            )
            d_jaccards.append(
                _topk_jaccard(base_d, replica_d, ego, core_budget, absolute=True)
            )
    return {
        "capacity": c_mean,
        "direction": d_mean,
        "min_c_topk_jaccard": float(min(c_jaccards)) if c_jaccards else 1.0,
        "min_absd_topk_jaccard": float(min(d_jaccards)) if d_jaccards else 1.0,
        "mean_c_topk_jaccard": float(np.mean(c_jaccards)) if c_jaccards else 1.0,
        "mean_absd_topk_jaccard": float(np.mean(d_jaccards)) if d_jaccards else 1.0,
    }

def _mean_oracle_capacity(oracle_bank):
    if not oracle_bank:
        raise ValueError("oracle capacity bank is empty")
    output = {}
    for ego in oracle_bank[0]:
        output[ego] = {
            source: float(np.mean([
                bank[ego][source] for bank in oracle_bank
            ]))
            for source in oracle_bank[0][ego]
        }
    return output


def _f1(predicted, truth):
    denom = len(predicted) + len(truth)
    return 2.0 * len(predicted & truth) / denom if denom else 1.0


def _capacity_ndcg_at_k(predicted_ranking, oracle_scores, core_budget):
    """nDCG retains the selector's rank order; never sort by oracle gain."""
    gains = [
        max(0.0, float(oracle_scores.get(int(j), 0.0)))
        for j in list(predicted_ranking)[:int(core_budget)]
    ]
    ideal = sorted(
        (max(0.0, float(value)) for value in oracle_scores.values()),
        reverse=True,
    )[:int(core_budget)]
    dcg = sum(gain / np.log2(index + 2.0) for index, gain in enumerate(gains))
    idcg = sum(gain / np.log2(index + 2.0) for index, gain in enumerate(ideal))
    return float(dcg / idcg) if idcg > 1e-12 else 1.0


def _pretrain_checkpoint(
    seed, episodes, core_budget, device, n_agents=24
):
    """Pretrain a selector-neutral full-explicit representation checkpoint."""
    RE.set_global_seed(seed)
    cfg = _cfg(seed, core_budget, selector="full_explicit")
    env = _make_env(
        seed, phase_length=max(100000, int(episodes) + 1), n_agents=n_agents
    )
    runner = RE.make_runner("Final-CIGAMF", env, cfg, device)
    for belief in runner.belief_modules.values():
        belief.max_core_size = len(belief.neighbor_ids)
        belief.set_fixed_core(belief.neighbor_ids)
    runner.pair_rel_module.reconcile_core_sets(
        {ego: belief.get_core_set() for ego, belief in runner.belief_modules.items()}
    )
    runner.run(n_episodes=int(episodes), eval_every=max(1, int(episodes)))
    checkpoint = _capture_frozen_learning_checkpoint(runner)
    checkpoint["episodes"] = int(episodes)
    checkpoint["checkpoint_role"] = "selector_neutral_full_explicit_pretrain"
    return checkpoint


def _initial_checkpoint(seed, core_budget, device, n_agents=24):
    """Capture common untrained initialization for end-to-end selector runs."""
    RE.set_global_seed(seed)
    cfg = _cfg(seed, core_budget, selector="structural_capacity")
    env = _make_env(seed, phase_length=100000, n_agents=n_agents)
    runner = RE.make_runner("Final-CIGAMF", env, cfg, device)
    checkpoint = _capture_frozen_learning_checkpoint(runner)
    checkpoint["episodes"] = 0
    checkpoint["checkpoint_role"] = "common_untrained_initialization"
    return checkpoint


def _isolation_rows(
    variants, seed, core_budget, device, checkpoint, state_bank, oracle_bank,
    oracle_direction_bank, teacher_contexts, n_agents=24,
):
    records = {variant: [] for variant in variants}
    for variant in variants:
        runner = _prepare_variant_runner(
            variant, seed, core_budget, device, checkpoint, n_agents=n_agents
        )
        runner.cfg["freeze_policy_learning"] = True
        runner.cfg["freeze_representation_state"] = True
        for state_index, (state, oracle_scores) in enumerate(
            zip(state_bank, oracle_bank)
        ):
            _restore_frozen_learning_checkpoint(runner, checkpoint)
            _apply_variant_allocation_state(
                runner, VARIANTS[variant], core_budget
            )
            runner.env.restore_state(copy.deepcopy(state))
            runner.env.set_behaviour_override("cooperative")
            runner.oracle_capacity_scores_by_ego = copy.deepcopy(oracle_scores)
            obs_all = runner.env._get_obs_all()
            # Every selector receives the *same* immutable factual co-actions
            # and target policy from the neutral reference runner.  Using a
            # variant's own first forward here makes the policy depend on the
            # allocation before the allocation itself is scored.
            common = teacher_contexts[int(state_index)]
            teacher_action_list = [int(value) for value in common["actions"]]
            runner._score_all_pairs_and_update_beliefs(
                obs_all=obs_all,
                actions=teacher_action_list,
                behaviour_probs=None,
                policy_probs=np.asarray(common["policy_probs"], dtype=np.float64),
            )
            _decision_actions, decision_cache = runner._select_actions_population(obs_all)
            # Fidelity is a deterministic property of the policy distribution,
            # not of epsilon-forcing/sampling noise.  Always derive the compared
            # action from the masked policy logits.
            actions = np.argmax(
                np.asarray(decision_cache["policy_logits"], dtype=np.float64), axis=-1
            ).astype(np.int64).tolist()
            for ego, belief in runner.belief_modules.items():
                predicted_ranking = sorted(
                    belief.get_core_set(),
                    key=lambda j: float(belief._lcb_score(int(j))),
                    reverse=True,
                )
                predicted = set(predicted_ranking)
                truth = _top_k(oracle_scores[int(ego)], ego, core_budget)
                records[variant].append({
                    "key": (int(state_index), int(ego)),
                    "predicted": predicted,
                    "truth": truth,
                    "f1": _f1(predicted, truth),
                    "ndcg": _capacity_ndcg_at_k(
                        predicted_ranking, oracle_scores[int(ego)], core_budget
                    ),
                    "core_size": len(predicted),
                    "logits": np.asarray(
                        decision_cache["policy_logits"], dtype=np.float64
                    ),
                    "values": np.asarray(
                        [decision_cache["value_cache"][agent] for agent in range(runner.n_agents)],
                        dtype=np.float64,
                    ),
                    "actions": np.asarray(actions, dtype=np.int64),
                })

    # The challenge subset is defined only from oracle C* and oracle D*, not
    # from the learned selectors under test. This prevents post-hoc filtering
    # on estimator success or failure.
    disagreement_keys = set()
    for state_index, (c_scores, d_scores) in enumerate(
        zip(oracle_bank, oracle_direction_bank)
    ):
        for ego in c_scores:
            c_top = _top_k(c_scores[int(ego)], ego, core_budget)
            d_top = _top_k(
                {j: abs(float(v)) for j, v in d_scores[int(ego)].items()},
                ego, core_budget,
            )
            if c_top != d_top:
                disagreement_keys.add((int(state_index), int(ego)))
    total_keys = len(records.get("C-Core", next(iter(records.values()), [])))
    rows = []
    reference_fidelity = {
        row["key"]: row for row in records.get("Full-Explicit", [])
    }
    for variant in variants:
        values = records[variant]
        selector = VARIANTS[variant]
        core_sizes = [row["core_size"] for row in values]
        if selector != "full_explicit" and any(
            size != int(core_budget) for size in core_sizes
        ):
            raise RuntimeError(
                f"{variant}/seed={seed} violated isolation budget: {core_sizes}"
            )
        disagreement = [
            row["f1"] for row in values if row["key"] in disagreement_keys
        ]
        fidelity = [
            _decision_fidelity(
                {
                    "logits": row["logits"][None, ...],
                    "values": row["values"][None, ...],
                    "actions": row["actions"][None, ...],
                },
                {
                    "logits": reference_fidelity[row["key"]]["logits"][None, ...],
                    "values": reference_fidelity[row["key"]]["values"][None, ...],
                    "actions": reference_fidelity[row["key"]]["actions"][None, ...],
                },
            )
            for row in values if row["key"] in reference_fidelity
        ]
        rows.append({
            "panel": "selector_isolation",
            "variant": variant,
            "selector": selector,
            "seed": int(seed),
            "episodes": 0,
            "pretrain_episodes": int(checkpoint["episodes"]),
            "checkpoint_sha256": checkpoint["sha256"],
            "core_budget": int(core_budget),
            "matched_budget": int(selector != "full_explicit"),
            "strict_5d_signature": True,
            "selector_state_count": int(len(state_bank)),
            "selector_pair_state_count": int(len(values)),
            "oracle_reference": "clone_state_all_action_fixed_rho_C",
            "mean_core_size": _mean(core_sizes),
            "selector_oracle_f1": _mean([row["f1"] for row in values]),
            "selector_oracle_ndcg_at_k": _mean([row["ndcg"] for row in values]),
            "disagreement_pair_state_count": int(len(disagreement_keys)),
            "disagreement_pair_state_fraction": float(
                len(disagreement_keys) / max(1, total_keys)
            ),
            "disagreement_selector_oracle_f1": _mean(disagreement),
            "mean_reward": float("nan"),
            "final_reward": float("nan"),
            "mean_f1": float("nan"),
            "throughput_total": float("nan"),
            "policy_logit_l2_to_full_explicit": _mean([
                item["policy_logit_l2_to_full_explicit"] for item in fidelity
            ]),
            "value_mae_to_full_explicit": _mean([
                item["value_mae_to_full_explicit"] for item in fidelity
            ]),
            "action_agreement_to_full_explicit": _mean([
                item["action_agreement_to_full_explicit"] for item in fidelity
            ]),
            "decision_fidelity_reference": "Full-Explicit",
            "decision_fidelity_protocol": (
                "common_frozen_policy_context_selector_then_forward"
            ),
            "checkpoint_role": checkpoint.get("checkpoint_role", "unspecified"),
        })
    return rows


def _end_to_end_row(
    variant, seed, episodes, core_budget, device, checkpoint,
    mean_oracle_capacity, n_agents=24,
):
    runner = _prepare_variant_runner(
        variant, seed, core_budget, device, checkpoint, n_agents=n_agents
    )
    if VARIANTS[variant] == "oracle_capacity":
        runner.oracle_capacity_scores_by_ego = copy.deepcopy(mean_oracle_capacity)
    history = runner.run(n_episodes=int(episodes), eval_every=10)
    core_sizes = [len(module.get_core_set()) for module in runner.belief_modules.values()]
    selector = VARIANTS[variant]
    if selector != "full_explicit" and any(size != int(core_budget) for size in core_sizes):
        raise RuntimeError(
            f"{variant}/seed={seed} violated matched budget: {core_sizes}"
        )
    row = {
        "panel": "end_to_end",
        "variant": variant,
        "selector": selector,
        "seed": int(seed),
        "episodes": int(episodes),
        "pretrain_episodes": int(checkpoint["episodes"]),
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_role": checkpoint.get("checkpoint_role", "unspecified"),
        "core_budget": int(core_budget),
        "matched_budget": int(selector != "full_explicit"),
        "strict_5d_signature": True,
        "mean_core_size": _mean(history.get("mean_core_size", [])),
        "selector_oracle_f1": float("nan"),
        "oracle_reference": (
            "static_bank_mean_oracle_capacity"
            if VARIANTS[variant] == "oracle_capacity" else "not_applicable"
        ),
        "mean_reward": _mean(history.get("mean_reward", [])),
        "final_reward": float(history.get("mean_reward", [float("nan")])[-1]),
        "mean_f1": _mean(history.get("mean_f1", [])),
        "throughput_total": _mean(history.get("throughput_total_agent_steps_per_sec", [])),
    }
    return row, _decision_probe(
        runner, n_states=4, seed=int(seed) + 89009
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--pretrain-episodes", type=int, default=60)
    parser.add_argument("--core-budget", type=int, default=3)
    parser.add_argument("--selector-states", type=int, default=8)
    parser.add_argument(
        "--challenge-oversample", type=int, default=3,
        help="Oracle-only oversampling factor used to construct C* vs |D*| disagreement states.",
    )
    parser.add_argument("--oracle-trials", type=int, default=8)
    parser.add_argument(
        "--oracle-replicates", type=int, default=2,
        help="Independent CRN oracle replicas used to gate top-k ranking stability.",
    )
    parser.add_argument(
        "--oracle-stability-min", type=float, default=0.80,
        help="Frozen minimum worst-ego top-k Jaccard for both C* and |D*|.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--agent-count", type=int, default=24)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    parser.add_argument("--out-root", default=os.path.join(ROOT, "results", "paper_b_allocation"))
    args = parser.parse_args(argv)
    if (
        args.episodes <= 0 or args.pretrain_episodes <= 0
        or args.selector_states <= 0 or args.challenge_oversample <= 0
        or args.oracle_trials < 2 or args.oracle_replicates < 2
        or not 0.0 <= args.oracle_stability_min <= 1.0
        or args.core_budget <= 0 or args.agent_count <= args.core_budget
        or not args.seeds
    ):
        parser.error(
            "episodes, pretrain episodes, core budget, and seeds must be positive"
        )
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    out_root = ensure_dir(os.path.abspath(args.out_root))
    rows = []
    selected_variants = args.variants or list(VARIANTS)
    for seed in args.seeds:
        isolation_checkpoint = _pretrain_checkpoint(
            seed, args.pretrain_episodes, args.core_budget, args.device,
            n_agents=args.agent_count
        )
        end_to_end_checkpoint = _initial_checkpoint(
            seed, args.core_budget, args.device, n_agents=args.agent_count
        )
        # Generate the immutable factual policy context from the selector-neutral
        # Full-Explicit checkpoint.  Using C-Core here would let the allocation
        # under test alter the very policy used to define D* and the common
        # co-action context before selectors are compared.
        reference_runner = _prepare_variant_runner(
            "Full-Explicit", seed, args.core_budget, args.device, isolation_checkpoint,
            n_agents=args.agent_count
        )
        reference_runner.env.set_behaviour_override("cooperative")
        bank_seed = int(seed) + 88001
        candidate_count = int(args.selector_states) * int(args.challenge_oversample)
        horizon = int(reference_runner.cfg["causal_horizon"])
        candidate_states = reference_runner.env.sample_state_bank(
            n_states=candidate_count, burn_in=3, bank_seed=bank_seed,
            min_remaining_steps=horizon,
        )
        discount = float(reference_runner.cfg["discount"])
        teacher_contexts_all = [
            _policy_context_for_state(reference_runner, state)
            for state in candidate_states
        ]
        candidate_oracles = []
        for state_index, state in enumerate(candidate_states):
            replicas = [
                _oracle_capacity_direction_for_state(
                    reference_runner.env, state, horizon=horizon, discount=discount,
                    trials=args.oracle_trials,
                    seed=(
                        bank_seed + state_index * 100003
                        + replica_index * 10000019
                    ),
                    target_policy_probs=teacher_contexts_all[state_index]["policy_probs"],
                )
                for replica_index in range(int(args.oracle_replicates))
            ]
            candidate_oracles.append(
                _oracle_replica_summary(replicas, args.core_budget)
            )
        stable_flags = [
            bool(
                item["min_c_topk_jaccard"] >= float(args.oracle_stability_min)
                and item["min_absd_topk_jaccard"] >= float(args.oracle_stability_min)
            )
            for item in candidate_oracles
        ]
        challenge_flags = []
        for item in candidate_oracles:
            c_scores, d_scores = item["capacity"], item["direction"]
            has_disagreement = any(
                _top_k(c_scores[ego], ego, args.core_budget) != _top_k(
                    {j: abs(float(v)) for j, v in d_scores[ego].items()},
                    ego, args.core_budget,
                )
                for ego in c_scores
            )
            challenge_flags.append(bool(has_disagreement))
        stable_indices = [index for index, flag in enumerate(stable_flags) if flag]
        if len(stable_indices) < int(args.selector_states):
            raise RuntimeError(
                "oracle selector bank failed the frozen ranking-stability gate: "
                f"stable={len(stable_indices)} required={int(args.selector_states)} "
                f"threshold={float(args.oracle_stability_min):.3f}"
            )
        selected_indices = sorted(
            stable_indices, key=lambda index: (not challenge_flags[index], index)
        )[:int(args.selector_states)]
        state_bank = [candidate_states[index] for index in selected_indices]
        oracle_bank = [candidate_oracles[index]["capacity"] for index in selected_indices]
        oracle_direction_bank = [
            candidate_oracles[index]["direction"] for index in selected_indices
        ]
        selected_stability = [candidate_oracles[index] for index in selected_indices]
        teacher_contexts = [teacher_contexts_all[index] for index in selected_indices]
        mean_oracle_capacity = _mean_oracle_capacity(oracle_bank)
        isolation_rows = _isolation_rows(
            selected_variants,
            seed,
            args.core_budget,
            args.device,
            isolation_checkpoint,
            state_bank,
            oracle_bank,
            oracle_direction_bank,
            teacher_contexts,
            n_agents=args.agent_count,
        )
        stability_summary = {
            "oracle_min_c_topk_jaccard": float(min(
                item["min_c_topk_jaccard"] for item in selected_stability
            )),
            "oracle_min_absd_topk_jaccard": float(min(
                item["min_absd_topk_jaccard"] for item in selected_stability
            )),
            "oracle_mean_c_topk_jaccard": float(np.mean([
                item["mean_c_topk_jaccard"] for item in selected_stability
            ])),
            "oracle_mean_absd_topk_jaccard": float(np.mean([
                item["mean_absd_topk_jaccard"] for item in selected_stability
            ])),
        }
        for row in isolation_rows:
            row.update(stability_summary)
        rows.extend(isolation_rows)
        end_to_end = {}
        for variant in selected_variants:
            end_to_end[variant] = _end_to_end_row(
                variant, seed, args.episodes, args.core_budget, args.device,
                end_to_end_checkpoint, mean_oracle_capacity,
                n_agents=args.agent_count,
            )
        reference = (
            end_to_end["Full-Explicit"][1]
            if "Full-Explicit" in end_to_end else None
        )
        for variant, (row, probe) in end_to_end.items():
            if reference is None:
                row.update({
                    "policy_logit_l2_to_full_explicit": float("nan"),
                    "value_mae_to_full_explicit": float("nan"),
                    "action_agreement_to_full_explicit": float("nan"),
                })
                row["decision_fidelity_reference"] = "not_collected"
            else:
                row.update(_decision_fidelity(probe, reference))
                row["decision_fidelity_reference"] = "Full-Explicit"
            rows.append(row)
    path = os.path.join(out_root, "summary_paper_b_allocation.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(os.path.join(out_root, "manifest.json"), {
        "experiment": "paper_b_c_vs_absd_allocation",
        "complete": True,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "pretrain_episodes": args.pretrain_episodes,
        "core_budget": args.core_budget,
        "agent_count": args.agent_count,
        "selector_states": args.selector_states,
        "challenge_oversample": args.challenge_oversample,
        "challenge_selected_state_count_by_seed": "see summary disagreement_pair_state_count",
        "oracle_trials": args.oracle_trials,
        "oracle_replicates": int(args.oracle_replicates),
        "oracle_stability_min": float(args.oracle_stability_min),
        "oracle_stability_gate": "worst-ego top-k Jaccard for C* and |D*|",
        "oracle_stable_candidate_count_by_seed": "fail-closed before state selection",
        "selector_bank_seed_rule": "training_seed_plus_88001",
        "selector_bank_held_out": True,
        "variants": {name: VARIANTS[name] for name in selected_variants},
        "panels": ["selector_isolation", "end_to_end"],
        "selector_isolation_checkpoint": "selector_neutral_full_explicit_pretrain",
        "end_to_end_checkpoint": "common_untrained_initialization",
        "summary": path,
    })
    print(path)


if __name__ == "__main__":
    main()
