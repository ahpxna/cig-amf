"""
code_test.py — Test the hypothesis that Phi is a per-step quantity while W*
is an H-step quantity, so their differing units explain the persistent
corr(Phi,W*) = 0.54–0.62 across every audit block, including BASELINE with
P1–P4 disabled.

Do not launch automatically. Run manually when requested:

    cd /path/to/cig_amf/envs
    python3 code_test.py                 # or ../cig-env/bin/python code_test.py

The script performs three tests while reusing the functions and data already
defined in env_audit.py rather than reimplementing oracle, Gini, or Pearson
logic:

  (1) Sweep H in {1,2,3,5,8}. Compute corr(Phi,W*) at every H over the same
      presampled state bank for an apples-to-apples comparison. High
      correlation at H=1 followed by a decline as H grows would confirm the
      "per-step Phi versus H-step W*" hypothesis.

  (2) Change the statistic. Rather than computing global Pearson across all
      (i,j) pairs and pooling egos with different reward scales, compute:
        - Spearman WITHIN EACH EGO, grouping by j as the influenced agent, and
          average over egos with at least three data points. Each collector ego
          receives four declared pairs—gatekeeper, relay, blocker, and
          controller—so it can be ranked. A gatekeeper ego receives only one
          pair from a collector, cannot be ranked, and is excluded and reported.
        - Sign agreement: the fraction of pairs for which
          sign(Phi) == sign(mean W*).
      If sign agreement is approximately 1.0 while Pearson remains near 0.6,
      the environment is not defective. The 0.70 gate is simply invalid for
      two quantities with different units. In that case, revise the criterion,
      not the environment.

  (3) Print gatekeeper->collector and blocker->collector examples across H for
      manual comparison with the causal hypothesis in the research write-up:
      opening a gate contributes +0.25 immediately and unlocks +1.0 later;
      blocking contributes -0.18 immediately and pushes +1.0 outside the
      horizon.

Structure value is intentionally not recomputed inside this H/Phi unit
diagnostic.  The repository now implements the separate zero-cost oracle-core
prerequisite in ``envs/structure_value_tier0.py`` and executes it through
``run_oracle.py``.  Keeping the experiments separate prevents this diagnostic
from fabricating or duplicating the Section-7 quantity.
"""
import sys
import numpy as np

try:
    from envs.env_audit import (
        OmniArena,
        gini,
        pearson_corr,
        oracle_w_star_sustained,
        build_declared_pair_list,
        sample_states,
        N_AGENTS, GRID_SIZE, N_ZONES, MAX_STEPS, PHASE_LENGTH, SEED,
    )
except ModuleNotFoundError:  # Allow ``python envs/code_test.py`` as well.
    from env_audit import (
        OmniArena,
        gini,
        pearson_corr,
        oracle_w_star_sustained,
        build_declared_pair_list,
        sample_states,
        N_AGENTS, GRID_SIZE, N_ZONES, MAX_STEPS, PHASE_LENGTH, SEED,
    )

H_VALUES = [1, 2, 3, 5, 8]
N_STATES = 10          # Shared states for every H (apples-to-apples).
SPEARMAN_MIN_N = 3     # Minimum points within one ego for Spearman.
SIGN_EPS = 1e-6        # Threshold treating Phi/W* as nonzero for sign tests.


# ------------------------------------------------------------------
# Manual Spearman implementation without a SciPy dependency.
# ------------------------------------------------------------------
def _rank_avg(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x, y):
    if len(x) < 2:
        return None
    return pearson_corr(_rank_avg(x), _rank_avg(y))


# ------------------------------------------------------------------
# Run the oracle at one specific H over a presampled state set.
# ------------------------------------------------------------------
def measure_at_horizon(env, declared_pairs, states, horizon):
    records = []  # (i, j, label, phi, mean_w, vals_per_state)
    for (i, j, label) in declared_pairs:
        phi = env.gt_influence_by_ego[j].get(i, 0.0)
        vals = []
        for st in states:
            env.restore_state(st)
            vals.append(
                oracle_w_star_sustained(env, ego=j, j=i, horizon=horizon, crn_seed=SEED)
            )
        records.append((i, j, label, phi, float(np.mean(vals)), vals))
    return records


def summarize(records):
    phis = [r[3] for r in records]
    ws = [r[4] for r in records]

    pearson = pearson_corr(phis, ws)

    # Spearman within each ego, grouped by j as the influenced agent.
    by_ego = {}
    for (i, j, label, phi, w, _vals) in records:
        by_ego.setdefault(j, []).append((phi, w))

    ego_spearmans = []
    ego_skipped = []
    for ego, pts in by_ego.items():
        if len(pts) >= SPEARMAN_MIN_N:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            rho = spearman_corr(xs, ys)
            if rho is not None:
                ego_spearmans.append(rho)
        else:
            ego_skipped.append((ego, len(pts)))

    mean_spearman = float(np.mean(ego_spearmans)) if ego_spearmans else None

    # Sign agreement over all declared pairs. Ignore phi=0 defensively even
    # though such pairs should not occur in declared_pairs.
    agree = 0
    total = 0
    for phi, w in zip(phis, ws):
        if abs(phi) < SIGN_EPS:
            continue
        total += 1
        if np.sign(phi) == np.sign(w):
            agree += 1
    sign_agreement = (agree / total) if total else None

    return {
        "pearson": pearson,
        "mean_spearman_per_ego": mean_spearman,
        "n_ego_qualified": len(ego_spearmans),
        "n_ego_skipped": ego_skipped,
        "sign_agreement": sign_agreement,
        "n_pairs_signed": total,
    }


def main():
    print("=" * 78)
    print("code_test.py -- H sweep and statistical test of Phi/W* unit mismatch")
    print("=" * 78)

    env = OmniArena(
        n_agents=N_AGENTS, grid_size=GRID_SIZE, n_zones=N_ZONES,
        max_steps=MAX_STEPS, phase_length=PHASE_LENGTH,
        causal_horizon=max(H_VALUES), mode="behavioral_drift", seed=SEED,
    )
    env.reset()

    declared_pairs = build_declared_pair_list(env)
    states = sample_states(env, N_STATES)

    print(f"\nconfig: n_agents={N_AGENTS} grid={GRID_SIZE} n_zones={N_ZONES} "
          f"n_states={N_STATES} n_declared_pairs={len(declared_pairs)} seed={SEED}")
    print(f"H sweep = {H_VALUES}")

    all_results = {}
    all_records = {}
    for H in H_VALUES:
        print(f"\n[H={H}] running oracle_w_star_sustained on {len(declared_pairs)} "
              f"declared pairs x {N_STATES} states ...")
        records = measure_at_horizon(env, declared_pairs, states, horizon=H)
        summ = summarize(records)
        all_results[H] = summ
        all_records[H] = records
        print(f"  Pearson corr(Phi,W*)              = {summ['pearson']:.4f}"
              if summ['pearson'] is not None else "  Pearson corr(Phi,W*) = N/A")
        if summ['mean_spearman_per_ego'] is not None:
            print(f"  mean Spearman per-ego (n_ego={summ['n_ego_qualified']}) = "
                  f"{summ['mean_spearman_per_ego']:.4f}")
        else:
            print(f"  mean Spearman per-ego = N/A (no ego has at least "
                  f"{SPEARMAN_MIN_N} points)")
        if summ['n_ego_skipped']:
            print(f"  skipped egos (insufficient points to rank): {summ['n_ego_skipped']}")
        if summ['sign_agreement'] is not None:
            print(f"  sign agreement                    = {summ['sign_agreement']:.4f} "
                  f"({summ['n_pairs_signed']} pairs with phi != 0)")

    # ------------------------------------------------------------
    # (1) H-sweep summary table that makes the trend directly visible.
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY TABLE -- H sweep")
    print("=" * 78)
    header = f"{'H':>4} | {'Pearson':>9} | {'mean Spearman/ego':>18} | {'sign agree':>10}"
    print(header)
    print("-" * len(header))
    for H in H_VALUES:
        s = all_results[H]
        p = f"{s['pearson']:.4f}" if s['pearson'] is not None else "N/A"
        sp = f"{s['mean_spearman_per_ego']:.4f}" if s['mean_spearman_per_ego'] is not None else "N/A"
        sa = f"{s['sign_agreement']:.4f}" if s['sign_agreement'] is not None else "N/A"
        print(f"{H:>4} | {p:>9} | {sp:>18} | {sa:>10}")

    pearson_series = [all_results[H]['pearson'] for H in H_VALUES]
    if all(v is not None for v in pearson_series):
        decreasing = all(pearson_series[k] >= pearson_series[k + 1] - 1e-9
                          for k in range(len(pearson_series) - 1))
        peak_at_h1 = pearson_series[0] == max(pearson_series)
        print(f"\nPearson at H=1 is the sweep maximum: {peak_at_h1}")
        print(f"Pearson is non-increasing with H:     {decreasing}")
        if peak_at_h1 and decreasing:
            print(">> HYPOTHESIS CONFIRMED: per-step Phi and H-step W* have "
                  "incompatible units. The environment is not defective; "
                  "redefine Phi as an H-step contribution rather than changing dynamics.")
        else:
            print(">> No clear decline with H was observed. This dataset DOES NOT "
                  "confirm the unit-mismatch hypothesis; inspect the raw values "
                  "before drawing a conclusion.")

    # ------------------------------------------------------------
    # (3) Manual examples across H: gatekeeper/blocker -> collector.
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("MANUAL CHECK -- gatekeeper->collector and blocker->collector")
    print("=" * 78)
    for label in ("gatekeeper->collector", "blocker->collector"):
        print(f"\n{label}:")
        for H in H_VALUES:
            recs = [r for r in all_records[H] if r[2] == label]
            if not recs:
                continue
            phi = recs[0][3]
            mean_w = float(np.mean([r[4] for r in recs]))
            print(f"  H={H:>2}  phi={phi:+.3f}  mean W* (averaged over {len(recs)} "
                  f"zones) = {mean_w:+.4f}")

    # ------------------------------------------------------------
    # (2)/(4) Interpret Pearson versus sign agreement and point to the
    # dedicated structure-value experiment.
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    last_h = H_VALUES[-1]
    s_last = all_results[last_h]
    if (s_last['sign_agreement'] is not None and s_last['sign_agreement'] >= 0.95
            and s_last['pearson'] is not None and s_last['pearson'] < 0.70):
        print(f"At H={last_h}: sign_agreement={s_last['sign_agreement']:.4f} (~1.0) "
              f"but Pearson={s_last['pearson']:.4f} (<0.70).")
        print(">> Ordering and sign are correct. Revise the CRITERION because "
              "the 0.70 threshold assumes quantities with the same units; do "
              "not change the ENVIRONMENT.")
    else:
        print(f"At H={last_h}: sign_agreement="
              f"{s_last['sign_agreement']}, Pearson={s_last['pearson']}. "
              "The 'good signs, weak Pearson' condition is not met; inspect "
              "the raw values above before drawing a conclusion.")

    print("\n[INFO] structure-value is evaluated separately by run_oracle.py "
          "using envs/structure_value_tier0.py. This H/Phi diagnostic does "
          "not duplicate that experiment or mix its gate into the unit check.")


if __name__ == "__main__":
    main()
