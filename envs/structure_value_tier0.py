"""Tier 0 oracle regret without training.

This experiment measures the theoretical value ceiling of structural
information: the benefit of knowing exactly what the core does versus knowing
only aggregate behavior.

Corrections from Tasks #11, #12, and #14:

- ``oracle_structure_regret`` compares k-step ego action sequences, with
  k_seq=2 by default, instead of one action. It can therefore detect strategies
  such as waiting one step before moving versus moving immediately.
- The broken ``_rollout_custom`` raised NameError for nonexistent seq/ego,
  ordered parameters incorrectly, and returned a scalar later indexed as a
  dictionary. ``_rollout_seq`` uses the real OmniArena/TinyOracleDIG API:
  clone/restore state, scripted_policy, and step. Neither ``forced_step`` nor
  ``rollout_seq`` exists on OmniArena, so actions are built explicitly as in
  ``oracle_w_star_sustained``. CRN infrastructure compares 36 sequences across
  q_full/q_blind on the same noise stream, reducing irrelevant variance.
- q_range normalization is contaminated by the worst never-selected sequence.
  Increasing any penalty can lower min Q, inflate q_range, and dilute normalized
  regret without making structure more valuable. q_range remains for backward
  comparison; std(Q)-based ``norm_by_std`` and changed-action statistics are
  primary.
"""

import itertools

import numpy as np

from omni_arena import OmniArena


def _rollout_seq(
    env, ego_id, seq, forced_others, horizon, discount, n_trials, rng,
    crn_seed=None,
):
    """Roll out an ego action sequence and return its scalar return.

    Force the ego to follow the length-k ``seq`` during the first k horizon
    steps, then return to scripted_policy. ``forced_others`` maps agents to
    actions forced throughout the horizon; q_blind uses it to replace the core
    with average-neighbor behavior, while q_full passes an empty mapping. Other
    agents always use scripted_policy. Actions are built explicitly because
    OmniArena has no forced_step/rollout_seq API.

    Return the ego-only discounted return averaged over n_trials. P0d CRN is
    shared across trials and branches to reduce intervention-unrelated variance.
    """
    saved = env.clone_state()
    k = len(seq)
    seed_base = int(crn_seed) if crn_seed is not None else int(env.seed)

    accs = []
    for trial in range(int(n_trials)):
        crn_rng = np.random.RandomState((seed_base * 9973 + trial) % (2**32))
        buffer = env._make_crn_buffer(horizon, crn_rng)

        env.restore_state(saved)
        env.set_noise_buffer(buffer)

        total, g = 0.0, 1.0
        done = False
        for t in range(int(horizon)):
            acts = [int(env.scripted_policy(i)) for i in range(env.n_agents)]
            if t < k:
                acts[ego_id] = int(seq[t])
            for j, a in forced_others.items():
                acts[j] = int(a)

            # Exp0 consumes scalar reward only. Avoid constructing observations
            # and info, especially w_by_pair, millions of times. These flags
            # affect return values only, preserving dynamics and the estimand.
            _, rew, done, _ = env.step(
                acts, return_obs=False, return_info=False
            )
            total += g * float(rew[ego_id])
            g *= discount
            if done:
                break

        env.clear_noise_buffer()
        accs.append(total)

    env.restore_state(saved)
    return float(np.mean(accs))


def _marginal_action_dist(env, n_samples=200, rng=None):
    rng = rng or np.random.RandomState(0)
    A = env.get_action_dim()
    counts = np.zeros(A, dtype=np.float64)

    saved = env.clone_state()
    for _ in range(n_samples):
        try:
            acts = [int(env.scripted_policy(i)) for i in range(env.n_agents)]
            for act in acts:
                counts[act] += 1.0
        except AttributeError:
            acts = [int(rng.randint(A)) for _ in range(env.n_agents)]
            for act in acts:
                counts[act] += 1.0
        env.step(acts, return_obs=False, return_info=False)
    env.restore_state(saved)

    return counts / max(1.0, counts.sum())


def oracle_structure_regret(
    env, ego_id, k_core=3, horizon=8, discount=0.95,
    n_trials=3, n_mf_samples=8, rng=None, k_seq=2,
    precomputed_core_ids=None, marginal_action_dist=None,
):
    rng = rng or np.random.RandomState(0)
    A = env.get_action_dim()
    N = env.n_agents
    saved = env.clone_state()

    # ---- 1. True core: top-k by |W*|. ----
    # intervention_action MUST be a real action such as env.STAY, not None.
    # With candidate_actions=None, the oracle starts from range(N_ACTIONS) and
    # appends an intervention absent from that range. None would become a fake
    # forced action and silently contaminate per_action/signed/range/best/worst.
    # env.STAY is already valid and adds no spurious candidate.
    if precomputed_core_ids is None:
        infl = {}
        for j in range(N):
            if j == ego_id:
                continue
            try:
                profile = env.compute_oracle_influence_from_current_state(
                    ego_id=ego_id, agent_j=j, intervention_action=env.STAY,
                    horizon=horizon, n_trials=n_trials,
                )
                infl[j] = abs(float(profile["signed"]))
            except Exception:
                infl[j] = 0.0
            env.restore_state(saved)

        core_ids = sorted(infl, key=infl.get, reverse=True)[:int(k_core)]
    else:
        core_ids = list(precomputed_core_ids)

    # ---- 2. Enumerate all k_seq-step sequences over A actions. ----
    # k_seq=2 and A=6 produce 36 sequences, matching Task #11.
    sequences = list(itertools.product(range(A), repeat=int(k_seq)))
    n_seq = len(sequences)

    # Share one CRN seed across all sequences and q_full/q_blind so all 36
    # candidates see the same resource-respawn noise stream.
    crn_seed = int(rng.randint(1, 2 ** 31 - 1))

    # ---- 3. Q_full(seq): full information without core replacement. ----
    q_full = np.zeros(n_seq, dtype=np.float64)
    for si, seq in enumerate(sequences):
        env.restore_state(saved)
        q_full[si] = _rollout_seq(
            env, ego_id, seq, forced_others={}, horizon=horizon,
            discount=discount, n_trials=n_trials, rng=rng, crn_seed=crn_seed,
        )

    # ---- 4. Q_blind(seq): replace the core with average-neighbor actions. ----
    marg = (
        np.asarray(marginal_action_dist, dtype=np.float64)
        if marginal_action_dist is not None
        else _marginal_action_dist(env, rng=rng)
    )
    q_blind = np.zeros(n_seq, dtype=np.float64)

    for si, seq in enumerate(sequences):
        vals = []
        for _ in range(int(n_mf_samples)):
            forced_others = {j: int(rng.choice(A, p=marg)) for j in core_ids}
            env.restore_state(saved)
            val = _rollout_seq(
                env, ego_id, seq, forced_others=forced_others, horizon=horizon,
                discount=discount, n_trials=1, rng=rng, crn_seed=crn_seed,
            )
            vals.append(val)
        q_blind[si] = float(np.mean(vals))

    env.restore_state(saved)

    a_full = int(np.argmax(q_full))
    a_blind = int(np.argmax(q_blind))
    # Regret is the true q_full value of the true optimum minus the true q_full
    # value of the sequence selected under structural blindness. This is the
    # actual opportunity cost of missing core information under real dynamics,
    # not under the blind model's own mean-field estimate.
    regret = float(q_full[a_full] - q_full[a_blind])

    q_std = float(np.std(q_full))
    q_median = float(np.median(q_full))

    return {
        "regret": regret,
        "q_best_full": float(q_full[a_full]),
        "q_chosen_blind": float(q_full[a_blind]),
        "q_range": float(q_full.max() - q_full.min()),
        "q_std": q_std,
        "q_max_minus_median": float(q_full.max() - q_median),
        "core_ids": core_ids,
        "best_seq_full": sequences[a_full],
        "best_seq_blind": sequences[a_blind],
        "action_changed": bool(a_full != a_blind),
    }


def _precompute_core_ids_all_egos(env, k_core, horizon, n_trials=3):
    """Compute the identical signed oracle for all egos while reusing rollouts.

    The legacy API repeated each (ego,j) call even though every rollout already
    returned all ego rewards. The all-egos API retains that vector, scans
    j-by-action, and averages actions to produce identical signed scores without
    repeating each trajectory 24 times.
    """
    A, N = env.get_action_dim(), env.n_agents
    influence = {ego: {} for ego in range(N)}
    snapshot = env.clone_state()

    for j in range(N):
        per_action = []
        for action in range(A):
            try:
                delta = env.compute_oracle_influence_all_egos_from_current_state(
                    agent_j=j,
                    intervention_action=action,
                    horizon=horizon,
                    n_trials=n_trials,
                )
            except Exception:
                delta = np.zeros(N, dtype=np.float64)
            per_action.append(np.asarray(delta, dtype=np.float64))
            env.restore_state(snapshot)

        signed = np.mean(np.stack(per_action, axis=0), axis=0)
        for ego in range(N):
            if ego != j:
                influence[ego][j] = abs(float(signed[ego]))

    env.restore_state(snapshot)
    return {
        ego: sorted(scores, key=scores.get, reverse=True)[:int(k_core)]
        for ego, scores in influence.items()
    }


def run_tier0(
    env, n_states=30, steps_between=8, k_core=3, horizon=8, egos=None,
    seed=0, k_seq=2,
):
    rng = np.random.RandomState(seed)
    egos = list(range(env.n_agents)) if egos is None else list(egos)

    rows = []
    env.reset()

    for s in range(int(n_states)):
        # Avoid crossing the episode boundary in the P2<->P4 trap.
        if env.t + horizon + steps_between >= env.max_steps:
            env.reset()

        for _ in range(int(steps_between)):
            env.step(
                [int(env.scripted_policy(i)) for i in range(env.n_agents)],
                return_obs=False, return_info=False,
            )

        # Both quantities depend on current state but not ego. Cache once per
        # state to preserve the estimand and remove thousands of duplicate
        # rollouts from full Exp0.
        state_snapshot = env.clone_state()
        core_ids_by_ego = _precompute_core_ids_all_egos(
            env, k_core=k_core, horizon=horizon, n_trials=3
        )
        env.restore_state(state_snapshot)
        marginal = _marginal_action_dist(env, rng=rng)
        env.restore_state(state_snapshot)

        print(
            f"[Exp0] state {s + 1}/{int(n_states)}: core cache ready; "
            f"evaluating {len(egos)} egos",
            flush=True,
        )

        for ego_idx, ego in enumerate(egos):
            rows.append(oracle_structure_regret(
                env, ego, k_core=k_core, horizon=horizon, rng=rng, k_seq=k_seq,
                precomputed_core_ids=core_ids_by_ego[int(ego)],
                marginal_action_dist=marginal,
            ))
            if (ego_idx + 1) % 4 == 0 or ego_idx + 1 == len(egos):
                print(
                    f"[Exp0] state {s + 1}/{int(n_states)}: "
                    f"ego {ego_idx + 1}/{len(egos)} complete",
                    flush=True,
                )

    regrets = np.array([r["regret"] for r in rows])
    ranges = np.array([r["q_range"] for r in rows])
    q_std = np.array([r["q_std"] for r in rows])
    changed = np.array([r["action_changed"] for r in rows])

    live = ranges > 1e-9
    live_std = q_std > 1e-9
    norm = float(np.mean(regrets[live] / ranges[live])) if live.any() else 0.0
    norm_by_std = (
        float(np.mean(regrets[live_std] / q_std[live_std]))
        if live_std.any() else float("nan")
    )

    mask = changed
    regret_given_changed = float(regrets[mask].mean()) if mask.any() else float("nan")

    mask_std = mask & live_std
    norm_given_changed_by_std = (
        float(np.mean(regrets[mask_std] / q_std[mask_std]))
        if mask_std.any() else float("nan")
    )

    p95_regret = float(np.percentile(regrets, 95)) if len(regrets) else float("nan")

    out = {
        "mean_regret": float(regrets.mean()),
        "median_regret": float(np.median(regrets)),
        "p90_regret": float(np.percentile(regrets, 90)),
        "p95_regret": p95_regret,
        "normalised_regret": norm,               # q_range based; contaminated as documented above.
        "norm_by_std": norm_by_std,               # std(Q) based; primary normalization.
        "frac_action_changed": float(changed.mean()),
        "regret_given_changed": regret_given_changed,
        "norm_given_changed_by_std": norm_given_changed_by_std,
        "n_samples": int(len(rows)),
    }

    def _fmt(x):
        return "NaN (không có mẫu action_changed)" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"

    print(f"""
TẦNG 0 — HỐI TIẾC ORACLE  (n={out['n_samples']}, k_seq={k_seq})
  Hối tiếc trung bình              = {out['mean_regret']:.4f}
  Hối tiếc trung vị                = {out['median_regret']:.4f}
  P90 hối tiếc                     = {out['p90_regret']:.4f}
  P95 hối tiếc                     = {out['p95_regret']:.4f}
  Hối tiếc chuẩn hoá (q_range)     = {out['normalised_regret']:.3f}   [LƯU Ý: mẫu số bị nhiễu bởi chuỗi TỆ NHẤT -- xem norm_by_std]
  Hối tiếc chuẩn hoá (std Q)       = {_fmt(out['norm_by_std'])}
  Tỷ lệ ĐỔI chuỗi hành động        = {out['frac_action_changed']:.1%}
  Hối tiếc TB | đã đổi             = {_fmt(out['regret_given_changed'])}
  Hối tiếc chuẩn hoá(std) | đã đổi = {_fmt(out['norm_given_changed_by_std'])}

  Đọc kết quả:
    Tỷ lệ đổi hành động < 10%  -> Cấu trúc gần như KHÔNG ảnh hưởng quyết định.
    Hối tiếc chuẩn hoá(std) < 0.05  -> Biết cấu trúc gần như vô ích ở đây.
    Cả hai đều cao             -> Cấu trúc đáng biết, chạy tiếp Tầng 1-2.
    (Nếu frac_action_changed == 0, các thống kê "| đã đổi" là NaN -- không
     phải lỗi, chỉ là không có mẫu nào để tính trên đó.)
""")
    return out, rows


if __name__ == "__main__":
    env = OmniArena(
        n_agents=24,
        grid_size=24,
        n_zones=4,
        enable_conditional_gates=True,
        enable_latency_ladder=True,
        enable_congestion=True,
        enable_structural_shift=True
    )

    print("Bắt đầu lấy mẫu Tầng 0 (dự kiến ~20 phút)...")
    out, rows = run_tier0(env=env, n_states=20, steps_between=10, k_core=3, horizon=8, seed=123)
