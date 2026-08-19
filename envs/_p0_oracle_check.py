"""
P0 acceptance check: fixed TinyOracleDIG oracle recovers hand-declared Phi
of DIG reasonably well (signed corr(W*, Phi) > 0.7), and no longer abs()es
away the sign (a Phi-negative pair must yield W*-negative).

Not part of the shipped package -- ad hoc script, safe to delete after use.
"""
import numpy as np

from tiny_oracle_dig import TinyOracleDIG


def pearsonr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    r = float(np.corrcoef(x, y)[0, 1])
    return r, None


def main():
    env = TinyOracleDIG(seed=7, causal_horizon=8)
    env.reset()

    # burn a few steps so agents are not all at their reset positions
    for _ in range(2):
        acts = [env.scripted_policy(i) for i in range(env.n_agents)]
        env.step(acts)

    phis = []
    w_stars = []
    pair_records = []

    for ego in range(env.n_agents ):
        gt = env.get_gt_influence_for_ego(ego)
        for j, phi in gt.items():
            if abs(phi) < 1e-9:
                continue
            # W* here = value ADDED by j's normal (scripted) behaviour,
            # measured against a neutral STAY intervention:
            #   W* = R(j scripted) - R(j forced STAY) = -(alt - base)
            # This is the natural orientation to compare against Phi, which
            # is declared as "how much does this neighbour's normal role
            # behaviour help/hurt ego". profile['signed'] itself is oriented
            # the other way (mean_a R(a) - R_base, i.e. effect of active
            # intervention vs baseline scripted behaviour) -- both are valid
            # readings of the same oracle, just different comparison points;
            # The STAY-referenced orientation is selected here because it
            # matches the quantity Phi was designed to encode.
            profile = env.compute_oracle_influence_from_current_state(
                ego_id=ego,
                agent_j=j,
                intervention_action=env.STAY,
                horizon=8,
                n_trials=1,
                forced_step=0,
                candidate_actions=[env.STAY],
            )
            w_star = -profile["signed"]
            phis.append(phi)
            w_stars.append(w_star)
            pair_records.append((ego, j, phi, w_star))

    phis = np.array(phis)
    w_stars = np.array(w_stars)

    rho, _ = pearsonr(phis, w_stars)
    print(f"n_pairs = {len(phis)}")
    print(f"signed corr(Phi, W*) = {rho:.4f}")

    neg_pairs = [(e, j, phi, w) for (e, j, phi, w) in pair_records if phi < 0]
    pos_pairs = [(e, j, phi, w) for (e, j, phi, w) in pair_records if phi > 0]
    print(f"n_negative_phi_pairs = {len(neg_pairs)}, mean W* on them = "
          f"{np.mean([w for *_, w in neg_pairs]):.4f}")
    print(f"n_positive_phi_pairs = {len(pos_pairs)}, mean W* on them = "
          f"{np.mean([w for *_, w in pos_pairs]):.4f}")

    print("\nsample pairs (ego, j, phi, W*_signed):")
    for rec in pair_records[:10]:
        print("  ", rec)

    ok_corr = rho > 0.7
    ok_sign = (np.mean([w for *_, w in neg_pairs]) < 0) and (np.mean([w for *_, w in pos_pairs]) > 0)
    print(f"\nPASS corr>0.7: {ok_corr}")
    print(f"PASS sign check (negative Phi -> negative W*, positive Phi -> positive W*): {ok_sign}")


if __name__ == "__main__":
    main()
