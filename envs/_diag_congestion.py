"""
_diag_congestion.py -- "5-phut" check truc tiep: r_emergent (P3/kenh [3],
tac nghen -- collision/lane-capacity/queue) co bao gio khac 0 khong, khi
chay scripted_policy tren OmniArena voi enable_congestion=True.

KHONG duoc tu chay trong phien lam viec nay (rang buoc cung: khong chay bat
ky script nao instantiate OmniArena va step() that -- xem yeu cau cua ban).
File nay CHI duoc VIET, ban tu chay tren may minh:

    cd /Users/phanan/cig_amf/envs
    python3 _diag_congestion.py          # (hoac ../cig-env/bin/python _diag_congestion.py)

BOI CANH (doc code, KHONG chay) -- xem bao cao day du kem theo, tom tat:
  - enable_congestion CO duoc noi voi reward that: gate collision penalty
    (step(), dong ~837-839: r_emergent[a] -= 0.015 khi occupancy>1) va
    lane-over-capacity/station-queue (dong ~923-943: r_emergent[a] -= 0.1
    hoac -= split), roi clip ve [-MAX_EMERGENT_MAGNITUDE, +MAX_EMERGENT_MAGNITUDE]
    voi MAX_EMERGENT_MAGNITUDE = 0.15 * min|phi_ij| = 0.15*0.25 = 0.0375
    (KHONG phai 0 -- khong thay bug "clip ve dung 0").
  - Khong thay dau hieu truncate float32 hay discount lam trit tieu ve 0
    (r_emergent la python float thuong, cast np.float32 chi o cho khac
    khong lien quan; discount gamma^h > 0 voi h huu han khong the ep ve 0).
  - GIA THUYET manh nhat sau khi doc code (CHUA xac nhan bang chay that):
    mat do 24 agent / grid 24x24 (~4%) + scripted_policy cho tung role di
    theo duong rieng bit (gate/resource/sink/lane_a/panel) khien dieu kien
    kich hoat (occupancy>1 tren 1 o, >2 agent trong ban kinh 1 cua 1 lane,
    >=2 agent trong ban kinh 1 cua resource) HIEM khi/khong bao gio dung
    trong 1 episode binh thuong -- TRU MOT NGOAI LE dang chu y: blocker
    CHU DONG duoi theo collector moi buoc (scripted_policy ROLE_BLOCKER =
    greedy(pos, positions[collector])), nen khi collector dung canh
    zone_resource de nhat resource, blocker rat co the cung nam trong ban
    kinh 1 cua resource -> co the du 2 agent de kich hoat "waiting queue"
    (dong ~932-939). Vi vay KHONG the loai tru hoan toan bang doc code
    tinh -- can log truc tiep de xac nhan.

Script nay lam dung 1 viec: chay N episode voi scripted_policy (+enable_congestion=True),
log r_emergent MOI BUOC, dem xem no co bao gio != 0 khong, va neu co thi
kenh nao (collision / lane / queue) kich hoat, agent nao, o buoc nao.
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
