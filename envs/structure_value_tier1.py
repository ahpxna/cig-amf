# structure_value_tier1.py
import numpy as np
import torch
import torch.nn as nn


def collect_dataset(env, n_episodes=40, horizon=8, discount=0.95,
                    k_core=3, seed=0):
    """Collect ``(obs_i, core_actions, mean_action, R_i^(H))`` samples.

    Three corrections follow the original result review where
    MSE_core > MSE_solo/mf and frac_from_core reached -88 trillion percent:

    (a) Data collection uses scripted_policy with 15% random noise, not a
        fully random policy. Under a fully random policy, gatekeepers almost
        never open gates at the required time, leaving no real structure to
        detect. Tier 0 uses scripted_policy and observes a signal; the original
        Tier 1 measured a materially different, effectively unstructured task.

    (b) The mean-field feature is tiled k_core times into ``mf_matched`` with
        the same k_core*A dimension as the core one-hot vector. This prevents
        the higher-dimensional core model from receiving free overfitting
        capacity in the MSE comparison.

    (c) Core membership is no longer measured once at mid-episode and reused
        for the full episode. Measured T3 CV=1.035 confirms strong state
        dependence through delta_ij(s), Section 2.1. Correct cores can differ
        across the beginning, middle, and end. Each episode is divided into
        thirds; an oracle evaluates a representative midpoint for each third,
        and every training sample uses the core for its actual segment.
    """
    rng = np.random.RandomState(seed)
    A = env.get_action_dim()
    N = env.n_agents

    X_obs, X_core, X_mf, Y = [], [], [], []

    for ep in range(n_episodes):
        env.reset()
        traj = []

        # Representative points near 1/6, 1/2, and 5/6 of max_steps cover the
        # three segments. Each is capped at max_steps-horizon so the oracle has
        # room for a complete rollout, satisfying the P2<->P4 trap guard
        # ``self.t + horizon <= self.max_steps``.
        safe_cap = max(0, env.max_steps - horizon)
        seg_targets = [
            min(safe_cap, max(0, int(env.max_steps * frac)))
            for frac in (1 / 6, 1 / 2, 5 / 6)
        ]
        snapshots = [None, None, None]

        for t in range(env.max_steps):
            acts = [int(env.scripted_policy(a)) if rng.rand() > 0.15
                    else int(rng.randint(A)) for a in range(N)]
            obs, rew, done, _ = env.step(acts)
            traj.append((obs, acts, rew))
            for si in range(3):
                if snapshots[si] is None and env.t >= seg_targets[si]:
                    snapshots[si] = env.clone_state()
            if done:
                break

        # Defensive handling for an episode ending before a target point:
        # reuse the nearest earlier snapshot, or the current state if none
        # exists. Fixed max_steps normally makes this path unreachable.
        last_valid = None
        for si in range(3):
            if snapshots[si] is None:
                snapshots[si] = last_valid if last_valid is not None else env.clone_state()
            else:
                last_valid = snapshots[si]

        # Determine the true core separately for each episode segment.
        core_by_ego_segs = []
        for si in range(3):
            env.restore_state(snapshots[si])
            core_by_ego = {}
            for ego in range(N):
                # intervention_action MUST be a real action such as env.STAY,
                # not None. The oracle appends values absent from
                # candidate_actions; None is outside range(N_ACTIONS), so it
                # becomes a fake rollout action with forced_action=None. This
                # does not crash, but silently contaminates per_action/signed.
                infl = {j: abs(float(env.compute_oracle_influence_from_current_state(
                            ego_id=ego, agent_j=j, intervention_action=env.STAY,
                            horizon=horizon, n_trials=1)))
                        for j in range(N) if j != ego}
                core_by_ego[ego] = sorted(infl, key=infl.get, reverse=True)[:k_core]
            core_by_ego_segs.append(core_by_ego)

        T = len(traj)
        seg_bounds = [0, T // 3, 2 * T // 3, T]

        for t in range(T):
            # Select the segment containing time step t.
            if t < seg_bounds[1]:
                seg_idx = 0
            elif t < seg_bounds[2]:
                seg_idx = 1
            else:
                seg_idx = 2
            core_by_ego = core_by_ego_segs[seg_idx]

            # Compute the H-step return.
            R = np.zeros(N)
            for h in range(horizon):
                if t + h < T:
                    R += (discount ** h) * np.asarray(traj[t + h][2])

            obs_t, acts_t, _ = traj[t]

            for ego in range(N):
                core = core_by_ego[ego]
                core_oh = np.zeros(k_core * A, dtype=np.float32)
                for idx, j in enumerate(core):
                    core_oh[idx * A + int(acts_t[j])] = 1.0

                others = [acts_t[j] for j in range(N) if j != ego]
                mf = np.bincount(others, minlength=A).astype(np.float32)
                mf /= max(1.0, mf.sum())
                # Tile mean-field to the core one-hot dimension (k_core*A) for
                # an MSE comparison with matched parameter capacity.
                mf_matched = np.tile(mf, k_core).astype(np.float32)

                X_obs.append(np.asarray(obs_t[ego], dtype=np.float32))
                X_core.append(core_oh)
                X_mf.append(mf_matched)
                Y.append(float(R[ego]))

    return (np.stack(X_obs), np.stack(X_core),
            np.stack(X_mf), np.asarray(Y, dtype=np.float32))


def _fit(X, Y, hidden=128, epochs=60, seed=0):
    """Fit an MLP regressor and return held-out MSE."""
    torch.manual_seed(seed)
    n = len(Y)
    idx = np.random.RandomState(seed).permutation(n)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]

    Xt = torch.tensor(X[tr]); Yt = torch.tensor(Y[tr])
    Xe = torch.tensor(X[te]); Ye = torch.tensor(Y[te])

    net = nn.Sequential(
        nn.Linear(X.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    for _ in range(epochs):
        perm = torch.randperm(len(Yt))
        for b in range(0, len(Yt), 256):
            sel = perm[b:b + 256]
            loss = nn.functional.mse_loss(net(Xt[sel]).squeeze(-1), Yt[sel])
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        return float(nn.functional.mse_loss(net(Xe).squeeze(-1), Ye))


def run_tier1(env, **kw):
    Xo, Xc, Xm, Y = collect_dataset(env, **kw)

    mse_core = _fit(np.concatenate([Xo, Xc], axis=1), Y)
    mse_mf   = _fit(np.concatenate([Xo, Xm], axis=1), Y)
    mse_solo = _fit(Xo, Y)

    # Measure how much neighbor information comes from identifying the core,
    # rather than knowing only aggregate crowd statistics.
    gain_total = mse_solo - mse_core       # all neighbour information
    gain_mf    = mse_solo - mse_mf         # information captured by mean field
    gain_core  = mse_mf - mse_core         # additional information from core identity

    # (d) frac=gain_core/gain_total can divide by a tiny or negative value
    # when the core predictor is worse than solo. The previous
    # max(1e-12, gain_total) discarded the real sign and produced meaningless
    # values such as -88 trillion percent. Report NaN when |gain_total| is too
    # small for a meaningful ratio.
    denom = gain_total if abs(gain_total) > 1e-9 else float("nan")
    frac = gain_core / denom

    # R² is the primary diagnostic. If all three values are near zero, none of
    # the models predicts the target, making MSE/frac comparisons meaningless;
    # their MSE differences then represent fit noise rather than signal.
    var_y = float(np.var(Y))
    r2 = lambda m: 1.0 - m / max(1e-12, var_y)
    r2_solo, r2_mf, r2_core = r2(mse_solo), r2(mse_mf), r2(mse_core)

    print(f"""
TẦNG 1 — KHOẢNG CÁCH DỰ ĐOÁN GIÁ TRỊ

  R² solo={r2_solo:.3f}  mf={r2_mf:.3f}  core={r2_core:.3f}
  (đọc dòng này TRƯỚC: nếu cả ba ≈ 0, không mô hình nào dự đoán được gì --
   MSE/gain/frac bên dưới không đáng tin cậy để so sánh.)

  MSE chỉ obs riêng          = {mse_solo:.5f}
  MSE + mean-field (matched) = {mse_mf:.5f}
  MSE + core đích danh       = {mse_core:.5f}

  thông tin do biết core     = {gain_core:.5f}
  phần trong tổng thông tin  = {frac:.1%} (NaN nếu |gain_total| <= 1e-9)

  frac < 10%  -> mean-field đã bắt gần hết; biết core thêm được ít.
  frac > 30%  -> biết đích danh core mang thông tin đáng kể.
""")
    return {"mse_solo": mse_solo, "mse_mf": mse_mf, "mse_core": mse_core,
            "r2_solo": r2_solo, "r2_mf": r2_mf, "r2_core": r2_core,
            "frac_from_core": float(frac)}

if __name__ == "__main__":
    # Import the environment directly when this file runs from envs.
    try:
        from omni_arena import OmniArena
    except ImportError:
        from envs.omni_arena import OmniArena

    # Enable conditional gates to create explicit structural dependence.
    env = OmniArena(
        enable_conditional_gates=True,
        enable_latency_ladder=False,
        enable_congestion=False,
        enable_structural_shift=False
    )

    print("======================================================================")
    print("ĐANG CHẠY ĐÁNH GIÁ TẦNG 1: KHOẢNG CÁCH DỰ ĐOÁN GIÁ TRỊ (PREDICTIVE)")
    print("======================================================================")
    print("Đang thu thập dữ liệu (Rollout)... Quá trình này có thể mất chút thời gian.")

    # Run Tier 1. Use 10-20 episodes only for a quicker code smoke run.
    results = run_tier1(
        env=env,
        n_episodes=40,    
        horizon=8,        
        discount=0.95,    
        k_core=2,         # Select the two strongest agents as the core.
        seed=123
    )
