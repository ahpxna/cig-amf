"""
code_test.py -- kiem tra gia thuyet "Phi la per-step, W* la H-step, hai dai
luong khac don vi" cho corr(Phi,W*) dang ket = 0.54-0.62 xuyen suot moi khoi
audit (ke ca BASELINE tat het P1-P4).

KHONG tu chay trong phien nay theo yeu cau -- ban chay tay tren may minh:

    cd /Users/phanan/cig_amf/envs
    python3 code_test.py                 # (hoac ../cig-env/bin/python code_test.py)

Ba phep thu, dung lai ham/so lieu co san trong env_audit.py (khong viet lai
logic oracle/gini/pearson):

  (1) Quet H in {1,2,3,5,8}: tinh corr(Phi,W*) tai moi H, cung mot tap state
      da sample truoc (apples-to-apples). Neu corr cao o H=1 roi tut dan theo
      H -> xac nhan gia thuyet "Phi per-step vs W* H-step".

  (2) Doi thong ke: thay vi Pearson toan cuc tren moi cap (i,j) (gop cac ego
      co thang reward khac nhau), tinh:
        - Spearman TRONG TUNG EGO (group theo j -- agent nhan anh huong) roi
          lay trung binh cac ego co >=3 diem du lieu (moi ego collector nhan
          4 cap khai bao: gatekeeper/relay/blocker/controller -> du de rank;
          ego gatekeeper chi nhan 1 cap tu collector -> khong du de rank, bi
          loai va bao cao rieng).
        - Sign agreement: ty le cap co sign(Phi) == sign(mean W*).
      Neu sign agreement ~ 1.0 ma Pearson van ~0.6 -> moi truong khong hong,
      chi la nguong 0.70 dat sai cho hai dai luong khac don vi (per-step vs
      H-step). Khi do sua tieu chi, khong sua moi truong.

  (3) In rieng vi du gatekeeper->collector va blocker->collector qua cac H
      de doi chieu bang tay voi gia thuyet nhan trong writeup cua ban (mo
      cong +0.25 mo khoa nhat +1.0 sau; block -0.18 day nhat +1.0 ra ngoai
      horizon).

Muc (4) trong yeu cau cua ban -- struct_value = reward(oracle-core) -
reward(pure-MF) -- CHUA duoc cai dat o dau trong repo nay (da grep
"structure_value" trong envs/, khong co ket qua). Khong bia so: script nay
CHỈ in canh bao o cuoi, khong tinh gia tri gia.
"""
import sys
import numpy as np

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
N_STATES = 10          # so state dung chung cho moi H (apples-to-apples)
SPEARMAN_MIN_N = 3      # so diem toi thieu trong 1 ego de tinh Spearman
SIGN_EPS = 1e-6          # nguong coi phi/W* la "khac 0" khi so dau


# ------------------------------------------------------------------
# Spearman thu cong (khong phu thuoc scipy)
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
# Chay oracle tai mot H cu the, tren mot tap state co san
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

    # Spearman trong tung ego (group theo j = agent nhan anh huong)
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

    # Sign agreement tren toan bo cap khai bao (bo qua cap phi=0, khong co
    # trong danh sach declared_pairs nhung phong thu)
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
    print("code_test.py -- quet H, doi thong ke, kiem tra gia thuyet don vi Phi vs W*")
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
            print(f"  mean Spearman per-ego = N/A (khong ego nao du "
                  f"{SPEARMAN_MIN_N} diem)")
        if summ['n_ego_skipped']:
            print(f"  ego bi bo qua (khong du diem de rank): {summ['n_ego_skipped']}")
        if summ['sign_agreement'] is not None:
            print(f"  sign agreement                    = {summ['sign_agreement']:.4f} "
                  f"({summ['n_pairs_signed']} cap co phi != 0)")

    # ------------------------------------------------------------
    # (1) Bang tong hop quet H -- de mat thay xu huong tang/giam
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BANG TONG HOP -- quet H")
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
        print(f"\nPearson tai H=1 la max trong day quet: {peak_at_h1}")
        print(f"Pearson giam dan (khong tang) theo H:    {decreasing}")
        if peak_at_h1 and decreasing:
            print(">> XAC NHAN gia thuyet: Phi (per-step) khong khop don vi voi "
                  "W* (H-step). Moi truong khong hong -- can khai bao lai Phi "
                  "theo dong gop H-buoc, khong sua dong luc moi truong.")
        else:
            print(">> KHONG thay mau hinh 'giam dan theo H' ro rang -- gia thuyet "
                  "don vi CHUA duoc xac nhan bang du lieu nay, can xem lai raw "
                  "numbers ben duoi truoc khi ket luan.")

    # ------------------------------------------------------------
    # (3) Vi du tay: gatekeeper->collector va blocker->collector qua cac H
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VI DU DOI CHIEU TAY -- gatekeeper->collector va blocker->collector")
    print("=" * 78)
    for label in ("gatekeeper->collector", "blocker->collector"):
        print(f"\n{label}:")
        for H in H_VALUES:
            recs = [r for r in all_records[H] if r[2] == label]
            if not recs:
                continue
            phi = recs[0][3]
            mean_w = float(np.mean([r[4] for r in recs]))
            print(f"  H={H:>2}  phi={phi:+.3f}  mean W* (trung binh {len(recs)} "
                  f"zone) = {mean_w:+.4f}")

    # ------------------------------------------------------------
    # (2)/(4) Doc ket luan Pearson vs sign-agreement, va canh bao structure_value
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DOC KET LUAN")
    print("=" * 78)
    last_h = H_VALUES[-1]
    s_last = all_results[last_h]
    if (s_last['sign_agreement'] is not None and s_last['sign_agreement'] >= 0.95
            and s_last['pearson'] is not None and s_last['pearson'] < 0.70):
        print(f"Tai H={last_h}: sign_agreement={s_last['sign_agreement']:.4f} (~1.0) "
              f"nhung Pearson={s_last['pearson']:.4f} (<0.70).")
        print(">> Dung thu tu va dung dau la chinh -- goi y sua TIEU CHI "
              "(nguong 0.70 dang danh cho hai dai luong cung don vi), khong "
              "sua MOI TRUONG.")
    else:
        print(f"Tai H={last_h}: sign_agreement="
              f"{s_last['sign_agreement']}, Pearson={s_last['pearson']}. "
              "Khong khop dieu kien 'sign tot, Pearson kem' -- xem lai raw "
              "numbers o tren truoc khi ket luan.")

    print("\n[CANH BAO] structure_value = reward(oracle-core) - reward(pure-MF) "
          "CHUA duoc cai dat trong repo nay (khong tim thay dinh nghia "
          "'structure_value' trong envs/). Script nay KHONG tinh gia tri gia "
          "cho no. Neu can so nay cho Phan 7, phai code rieng: (a) mot policy "
          "chi dung oracle-core pairs (top-k |W*|) de quyet dinh hanh dong, "
          "(b) mot policy pure mean-field/random lam doi chung, roi so sanh "
          "tong reward tren cung mot tap episode/seed.")


if __name__ == "__main__":
    main()
