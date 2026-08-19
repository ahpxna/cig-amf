"""Direct diagnostic for the P3 emergent-congestion reward channel.

The original observation was that r_emergent, channel [3] for collisions,
lane capacity, and station queues, appeared to remain zero under
scripted_policy with enable_congestion=True.

Code inspection established the following facts before measurement:

- enable_congestion does affect reward. A gate collision subtracts 0.015 when
  occupancy exceeds one, while lane-over-capacity and station-queue events
  subtract 0.1 or a split penalty. The result is clipped to
  [-MAX_EMERGENT_MAGNITUDE, +MAX_EMERGENT_MAGNITUDE], where
  MAX_EMERGENT_MAGNITUDE = 0.15 * min|phi_ij| = 0.15 * 0.25 = 0.0375.
  The clip bound is therefore nonzero.
- Neither float32 conversion nor discounting explains an exact zero.
  r_emergent is an ordinary Python float at this point, and gamma**h remains
  positive for finite h.
- The leading hypothesis was geometric rarity. With 24 agents on a 24x24 grid
  (approximately 4% density) and role-specific scripted paths toward separate
  gates, resources, sinks, lanes, and panels, activation conditions such as
  occupancy > 1 in one cell, more than two agents within radius one of a lane,
  or at least two agents within radius one of a resource may almost never occur.
- A notable exception prevents resolving the question from code inspection
  alone: ROLE_BLOCKER actively follows a collector every step. When the
  collector approaches zone_resource, the blocker may also enter radius one
  and activate the station queue condition.

The diagnostic resolves the question empirically. It runs N scripted episodes
with congestion enabled, records r_emergent at every step, counts nonzero
events, and reports the affected agent and timestep. It is included in the
full validation manifest in scripts/run_all.sh.
"""
import numpy as np

from omni_arena import OmniArena

N_EPISODES = 5
SEED = 123


def main():
    env = OmniArena(
        n_agents=24, grid_size=24, n_zones=4, max_steps=60, phase_length=6,
        causal_horizon=8, mode="behavioral_drift", seed=SEED,
        enable_conditional_gates=True, enable_latency_ladder=True,
        enable_congestion=True, enable_structural_shift=False,
    )

    n_steps_total = 0
    n_steps_with_nonzero_remergent = 0
    max_abs_remergent = 0.0
    nonzero_examples = []  # (episode, t, agent, r_emergent_value)

    for ep in range(N_EPISODES):
        env.reset()
        done = False
        t = 0
        while not done:
            acts = [env.scripted_policy(i) for i in range(env.n_agents)]
            _, rew, done, info = env.step(acts)
            r_emergent = info["r_emergent"]
            n_steps_total += 1
            nz = [(a, v) for a, v in enumerate(r_emergent) if abs(v) > 1e-12]
            if nz:
                n_steps_with_nonzero_remergent += 1
                for a, v in nz:
                    max_abs_remergent = max(max_abs_remergent, abs(v))
                    if len(nonzero_examples) < 20:
                        nonzero_examples.append((ep, t, a, v))
            t += 1

    print("=" * 70)
    print("DIAG: r_emergent (P3 congestion channel) trực tiếp mỗi bước")
    print("=" * 70)
    print(f"n_episodes={N_EPISODES}  total_steps={n_steps_total}")
    print(f"steps with ANY agent having r_emergent != 0: "
          f"{n_steps_with_nonzero_remergent} / {n_steps_total} "
          f"({100.0 * n_steps_with_nonzero_remergent / max(1, n_steps_total):.2f}%)")
    print(f"max |r_emergent| observed = {max_abs_remergent:.6f} "
          f"(cap = MAX_EMERGENT_MAGNITUDE = {env.MAX_EMERGENT_MAGNITUDE:.6f})")

    if nonzero_examples:
        print("\nfirst nonzero examples (episode, t, agent, r_emergent):")
        for ex in nonzero_examples:
            print(f"  {ex}")
        print("\n>> r_emergent DOES trigger under scripted_policy -- hypothesis 3 "
              "(density too low) is REFUTED by this run; periphery channel is "
              "not empty, though it may still be rare/small in aggregate.")
    else:
        print("\n>> r_emergent was EXACTLY 0 on every single step across all "
              f"{N_EPISODES} episodes -- CONFIRMS hypothesis 3 (congestion "
              "trigger conditions never satisfied at this density under "
              "scripted_policy). This matches the code-reading prediction: "
              "collision/lane/queue trigger conditions are geometrically rare "
              "given ~4% agent density and role-specific disjoint paths.")


if __name__ == "__main__":
    main()
