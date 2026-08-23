"""[C5 — spec part C5] Strongest figure available, following SGTP Fig. 4 model:
W*_ij(t) HAS SIGN according to time within ONE episode.

  - one line per pair; declared lines are bold, control lines are light
  - annotate phase (phase of behavioural drift / structural shift milestones)
  - include a lower panel: number of pairs in close-coupling (C4) over time

This figure answers three questions in one view:
  1. does control separate from 0?            -> T5 / std|W*|(control)
  2. does influence turn on/off with state?   -> T3 / T6
  3. is influence continuous or only at rare tail? -> Tier-0 vs Tier-1 question

Run BEFORE and AFTER C1-C3 and place side by side: the "before" version produces flat lines at 0 plus two staircase lines — that is the intuitive evidence for the entire RC-1..RC-5 diagnosis.

  python scripts/plot_w_star_timeline.py                    # after (SGTP phi)
  python scripts/plot_w_star_timeline.py --no-sgtp --tag before"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

from omni_arena import OmniArena          # noqa: E402
import env_audit as EA                    # noqa: E402


def collect(env, horizon, n_steps, max_declared, max_control, seed):
    rng = np.random.RandomState(seed)
    declared = EA.build_declared_pair_list(env)[:max_declared]
    declared_set = {(i, j) for (i, j, *_r) in
                    [(d[0], d[1]) for d in declared]}
    control = EA.build_control_pair_list(env, declared_set, rng,
                                         n=max_control)

    pairs = ([(int(d[0]), int(d[1]), "declared") for d in declared]
             + [(int(c[0]), int(c[1]), "control") for c in control])

    series = {(i, j): [] for (i, j, _k) in pairs}
    csd, phases = [], []

    for t in range(n_steps):
        # Guard B: oracle rollout must not be cut across episode boundary.
        if env.t + horizon > env.max_steps:
            break
        for (i, j, _kind) in pairs:
            series[(i, j)].append(
                EA.oracle_w_star_sustained(env, ego=j, j=i, horizon=horizon)
            )
        csd.append(env.close_coupling_pairs()
                   if hasattr(env, "close_coupling_pairs") else 0)
        phases.append(int(getattr(env, "current_phase", 0)))
        env.step([env.scripted_policy(a) for a in range(env.n_agents)])

    return pairs, series, csd, phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sgtp", action="store_true",
                    help="use the legacy Phi lookup table")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--declared", type=int, default=8)
    ap.add_argument("--control", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(ROOT, "results", "figures"),
        help="Directory for the figure; defaults to results/figures.",
    )
    a = ap.parse_args()

    tag = a.tag or ("before" if a.no_sgtp else "after")
    env = OmniArena(n_agents=24, grid_size=24, n_zones=4, seed=a.seed,
                    use_sgtp_phi=not a.no_sgtp)
    env.reset()

    pairs, series, csd, phases = collect(
        env, a.horizon, a.steps, a.declared, a.control, a.seed)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    for (i, j, kind) in pairs:
        y = series[(i, j)]
        if not y:
            continue
        x = np.arange(len(y))
        if kind == "declared":
            ax.plot(x, y, lw=2.0, alpha=0.95,
                    label=f"{i}->{j} ({env.agent_role[i]}->{env.agent_role[j]})")
        else:
            ax.plot(x, y, lw=0.8, alpha=0.35, color="grey")

    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("$W^*_{ij}(t)$  (signed)")
    ax.set_title(
        f"Signed oracle influence over one episode — {tag}"
        "  (dark = declared, light = control)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    # annotate phase boundary (behavioural drift) — following SGTP Fig. 4 model
    for t in range(1, len(phases)):
        if phases[t] != phases[t - 1]:
            ax.axvline(t, color="tab:red", lw=1.0, ls=":")
            ax.text(t, ax.get_ylim()[1], f"phase {phases[t]}",
                    fontsize=7, color="tab:red", rotation=90, va="top")

    ax2.plot(np.arange(len(csd)), csd, color="tab:blue", lw=1.5)
    ax2.set_ylabel("# close-\ncoupling pairs")
    ax2.set_xlabel("timestep trong episode")
    ax2.grid(alpha=0.3)

    out_dir = os.path.abspath(os.path.expanduser(a.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"w_star_timeline_{tag}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"[C5] saved {out}")

    # data accompanying the figure — to avoid having to "read by eye"
    dec = np.array([v for (i, j, k) in pairs if k == "declared"
                    for v in series[(i, j)]], dtype=float)
    ctl = np.array([v for (i, j, k) in pairs if k == "control"
                    for v in series[(i, j)]], dtype=float)
    if ctl.size:
        print(f"[C5] control  mean|W*|={np.abs(ctl).mean():.4f} "
              f"std|W*|={np.abs(ctl).std():.4f} "
              f"std/mean={np.abs(ctl).std()/max(np.abs(ctl).mean(),1e-9):.2f} "
              f"(gate S5: 0.3-0.6)")
    if dec.size:
        print(f"[C5] declared mean|W*|={np.abs(dec).mean():.4f}")
    if csd:
        print(f"[C5] CSD mean close-coupling pairs = {np.mean(csd):.2f}")


if __name__ == "__main__":
    main()
