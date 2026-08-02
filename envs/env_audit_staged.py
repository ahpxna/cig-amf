"""
env_audit_staged.py — kiểm định Omni-Arena theo 3 khối tích luỹ (staged),
thay vì bật hết P1-P4 cùng lúc như env_audit.py.

LÝ DO (xem báo cáo triển khai của người dùng, không lặp lại toàn văn ở đây):
P1-P4 có thể triệt tiêu lẫn nhau một cách im lặng khi build+test chung một
lần -- P3 (congestion) có thể nhấn chìm P1 (conditional influence) làm T3
sụp; nhiễu hàng đợi của P3 trộn với độ trễ thiết kế của P2 làm T4 "đẹp"
nhầm lý do; cửa sổ oracle H=8 của P2 có thể trung bình hoá xuyên hai cấu
trúc nếu P4 xảy ra giữa chừng; P4 dời bottleneck đổi đúng các biến trạng
thái mà delta_ij(s) của P1 đọc.

Cấu trúc tích luỹ, THEO QUYẾT ĐỊNH CUỐI CỦA NGƯỜI DÙNG (P1+P2 gộp chung một
khối vì cả hai không đụng tới environment dynamics, chỉ đụng bảng ảnh
hưởng, và không tương tác với nhau):

  BASELINE  P0 only (mọi cờ P1-P4 tắt)               -- điểm mốc để so Block A
  Block A   P0 + P1 + P2  (conditional_gates=True, latency_ladder=True,
                            congestion=False, structural_shift=False)
  Block B   Block A + P3  (congestion=True)           -- MANDATORY, phải cô lập
  Block C   Block B + P4  (structural_shift=True)

Script này KHÔNG thay thế env_audit.py (env_audit.py = "bật hết cùng lúc",
vẫn hữu ích làm kiểm tra cuối cùng "mọi cờ bật"). Mọi hàm tính metric (T1-T6,
corr(Phi,W*)) được IMPORT từ env_audit.py, không viết lại logic.
"""
import sys
import numpy as np

from omni_arena import OmniArena
import env_audit as ea  # reuse metric-computation functions, no forking


# ============================================================
# Config -- giống env_audit.py để số liệu so sánh được với nhau
# ============================================================
BASE_ENV_KWARGS = dict(
    n_agents=ea.N_AGENTS,
    grid_size=ea.GRID_SIZE,
    n_zones=ea.N_ZONES,
    max_steps=ea.MAX_STEPS,
    phase_length=ea.PHASE_LENGTH,
    causal_horizon=ea.HORIZON,
)
SEED = ea.SEED

# ------------------------------------------------------------------
# Cờ P1-P4 cho từng khối. BASELINE không nằm trong 3 khối chính thức
# (A/B/C) -- nó chỉ là điểm mốc "P0 only" để Block A có cái để so delta,
# đúng tinh thần "T3 nên đi từ baseline DIG-như (yếu) lên PASS" trong đặc tả.
# ------------------------------------------------------------------
BLOCKS = [
    (
        "BASELINE",
        "P0 only (mọi cờ P1-P4 tắt) -- điểm mốc, KHÔNG phải 1 trong 3 khối chính thức",
        dict(
            enable_conditional_gates=False,
            enable_latency_ladder=False,
            enable_congestion=False,
            enable_structural_shift=False,
        ),
    ),
    (
        "A",
        "P0 + P1 + P2 (conditional_gates, latency_ladder) -- rủi ro THẤP, chỉ đụng bảng ảnh hưởng",
        dict(
            enable_conditional_gates=True,
            enable_latency_ladder=True,
            enable_congestion=False,
            enable_structural_shift=False,
        ),
    ),
    (
        "B",
        "A + P3 (congestion) -- rủi ro CAO, có thể phá T1 -- MANDATORY, phải cô lập",
        dict(
            enable_conditional_gates=True,
            enable_latency_ladder=True,
            enable_congestion=True,
            enable_structural_shift=False,
        ),
    ),
    (
        "C",
        "B + P4 (structural_shift) -- rủi ro TRUNG BÌNH, tương tác với H",
        dict(
            enable_conditional_gates=True,
            enable_latency_ladder=True,
            enable_congestion=True,
            enable_structural_shift=True,
        ),
    ),
]


def run_block(flags):
    """
    Chạy đầy đủ bộ metric T1-T6 + corr(Phi,W*) cho MỘT khối, dùng đúng các
    hàm của env_audit.py (không viết lại logic). Env chính (T1/T2/T3/T4/T5/
    corr/sign-checks) luôn mode="behavioral_drift" -- giống hệt cách
    env_audit.py.main() làm -- vì structural shift chỉ được thăm dò riêng
    qua các env chuyên dụng bên trong ea.run_t6().
    """
    env_kwargs = dict(BASE_ENV_KWARGS, **flags)
    main_env = OmniArena(mode="behavioral_drift", seed=SEED, **env_kwargs)
    main_env.reset()

    m1 = ea.run_oracle_based_metrics(main_env)
    m3 = ea.run_t3(main_env)
    m4 = ea.run_t4(main_env)
    m6 = ea.run_t6(env_kwargs=dict(env_kwargs), seed=SEED)
    msign = ea.run_oracle_sign_check(main_env)
    mabs = ea.oracle_no_abs_direct_check(main_env)

    return {
        "t1_gini": m1["t1_gini"],
        "t2_neg_frac": m1["t2_neg_frac"],
        "t2_pos_frac": m1["t2_pos_frac"],
        "t3_cv_mean": m3["t3_cv_mean"],
        "t4_spread": m4["t4_spread"],
        "t5_snr": m1["t5_snr"],
        "t6_ratio": m6["t6_ratio"],
        "t6_behav_exact": m6["t6_delta_phi_behavioural"] == 0.0,
        "corr_phi_w": m1["corr_phi_w"],
        "_raw": {"m1": m1, "m3": m3, "m4": m4, "m6": m6, "msign": msign, "mabs": mabs},
    }


METRIC_ORDER = [
    # [docs/CIG-AMF_training_debug_master.md mục 5.5] t5_snr/t6_ratio/
    # corr_phi_w giờ có CHẶN TRÊN, không chỉ chặn dưới. Trước đây audit chỉ
    # hỏi "tín hiệu có tồn tại không" (v > threshold) -- một SNR=1e9 pass
    # gate đó dễ dàng, nhưng SNR=1e9 chỉ chứng minh mày đang hard-code một
    # bảng tra cứu rồi đo lại chính giả định của mình (control pairs
    # W*=0 tuyệt đối), không chứng minh env có cấu trúc "phát hiện được
    # nhưng không tầm thường". Đổi sang khoảng [3,20] (SNR/T6) và
    # [0.70,0.95] (corr) biến audit thành "bài toán không tầm thường",
    # đúng thay đổi quan trọng nhất trong bảng gate của debug doc.
    ("t1_gini", "T1 Gini(|W*|)", lambda v: v > 0.30),
    ("t2_neg_frac", "T2 neg frac", lambda v: v > 0.15),
    ("t2_pos_frac", "T2 pos frac", lambda v: v > 0.15),
    ("t3_cv_mean", "T3 mean CV", lambda v: v > 0.30),
    ("t4_spread", "T4 sign-flip frac (NEW methodology)", lambda v: v > 0.3),
    ("t5_snr", "T5 SNR", lambda v: 3.0 <= v <= 20.0),
    ("t6_ratio", "T6 ratio", lambda v: 3.0 <= v <= 20.0),
    ("corr_phi_w", "corr(Phi,W*)", lambda v: 0.70 <= v <= 0.95),
]


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if abs(v) >= 1e6:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def print_block_result(name, desc, flags, metrics, prev_metrics, prev_name):
    print("\n" + "=" * 78)
    print(f"BLOCK {name}: {desc}")
    print("=" * 78)
    print(
        f"flags: enable_conditional_gates={flags['enable_conditional_gates']}  "
        f"enable_latency_ladder={flags['enable_latency_ladder']}  "
        f"enable_congestion={flags['enable_congestion']}  "
        f"enable_structural_shift={flags['enable_structural_shift']}"
    )

    name_w = max(len(label) for _, label, _ in METRIC_ORDER)
    for key, label, passfn in METRIC_ORDER:
        val = metrics[key]
        status = "PASS" if passfn(val) else "FAIL"
        line = f"  [{status}] {label:{name_w}s}  value = {fmt(val)}"
        if prev_metrics is not None:
            prev_val = prev_metrics[key]
            delta = val - prev_val
            prev_status = "PASS" if passfn(prev_val) else "FAIL"
            arrow = f"{fmt(prev_val)} -> {fmt(val)} (delta{delta:+.4f}, was {prev_status} now {status})"
            line += f"    [{prev_name}->{name}: {arrow}]"
        print(line)

    print(
        f"  [T6 behavioural invariance exact (||dPhi||=0)] = "
        f"{metrics['t6_behav_exact']}"
    )
    print(
        f"  [oracle no-abs() has_negative_delta] = "
        f"{metrics['_raw']['mabs']['has_negative_delta']}"
    )


def main():
    print("#" * 78)
    print("# OMNI-ARENA env_audit_staged.py -- 3 cumulative blocks (A, B, C)")
    print("# Reuses T1-T6/corr metric functions from env_audit.py verbatim.")
    print("#" * 78)
    print(
        f"\nshared config: n_agents={BASE_ENV_KWARGS['n_agents']} "
        f"grid={BASE_ENV_KWARGS['grid_size']} n_zones={BASE_ENV_KWARGS['n_zones']} "
        f"H={BASE_ENV_KWARGS['causal_horizon']} n_states_T1={ea.N_STATES_T1} "
        f"n_states_T3={ea.N_STATES_T3} n_control_pairs={ea.N_CONTROL_PAIRS} "
        f"n_trials={ea.N_TRIALS} seed={SEED}"
    )

    results = {}
    prev_name = None
    prev_metrics = None
    order = []

    for name, desc, flags in BLOCKS:
        print(f"\n>>> running block {name} ...")
        metrics = run_block(flags)
        results[name] = (desc, flags, metrics)
        order.append(name)
        print_block_result(name, desc, flags, metrics, prev_metrics, prev_name)
        prev_name, prev_metrics = name, metrics

    # ------------------------------------------------------------
    # Diagnostic call-outs per user's expected results
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DIAGNOSTIC CALL-OUTS (per staged-audit plan)")
    print("=" * 78)

    base_m = results["BASELINE"][2]
    a_m = results["A"][2]
    b_m = results["B"][2]
    c_m = results["C"][2]

    print(
        f"\n[Block A] T3 mean CV: BASELINE={fmt(base_m['t3_cv_mean'])} -> "
        f"A={fmt(a_m['t3_cv_mean'])}  "
        f"({'PASS' if a_m['t3_cv_mean'] > 0.30 else 'FAIL'}, expect BASELINE weak -> A PASS)"
    )
    print(
        f"[Block A] T4 sign-flip role frac (NEW methodology -- fixed "
        f"forced_step=0, H-sweep {ea.T4_H_SWEEP}, see env_audit.py's run_t4() "
        f"docstring; key 't4_spread' kept for compat, meaning changed from the "
        f"old sliding-forced_step 'peak spread'): "
        f"BASELINE={fmt(base_m['t4_spread'])} -> A={fmt(a_m['t4_spread'])}  "
        f"({'PASS' if a_m['t4_spread'] > 0.3 else 'FAIL'}, expect BASELINE weak -> A PASS)"
    )
    print(
        f"[Block A] T1/T2/T5 should already be fine at baseline (P1/P2 don't "
        f"touch congestion): "
        f"T1 BASELINE={fmt(base_m['t1_gini'])} A={fmt(a_m['t1_gini'])}; "
        f"T5 BASELINE={fmt(base_m['t5_snr'])} A={fmt(a_m['t5_snr'])}"
    )

    t1_b_verdict = (
        "STILL PASS -- OK" if b_m["t1_gini"] > 0.30 else
        "DROPPED BELOW 0.30 -- per blueprint Sec 1.2, HALVE the P3 congestion "
        "amplitude cap (0.15x -> 0.075x) and re-run Block B"
    )
    print(
        f"\n[Block B -- CRITICAL] T1 Gini MUST STAY PASS (>0.30): "
        f"A={fmt(a_m['t1_gini'])} -> B={fmt(b_m['t1_gini'])}  ({t1_b_verdict})"
    )
    print(
        f"[Block B] T3 mean CV (checking the exact P3<->P1 interaction risk -- "
        f"does congestion swamp the blocker's conditional gate?): "
        f"A={fmt(a_m['t3_cv_mean'])} -> B={fmt(b_m['t3_cv_mean'])}  "
        f"({'held up' if b_m['t3_cv_mean'] > 0.30 else 'DEGRADED below 0.30 threshold'})"
    )

    print(
        f"\n[Block C] T6 ratio: B={fmt(b_m['t6_ratio'])} -> C={fmt(c_m['t6_ratio'])}  "
        f"({'PASS (>3.0)' if c_m['t6_ratio'] > 3.0 else 'FAIL (<=3.0)'})"
    )
    print(
        f"[Block C] ||dPhi||(behavioural) exactly 0: {c_m['t6_behav_exact']}  "
        f"({'PASS' if c_m['t6_behav_exact'] else 'FAIL -- leak between modes, per blueprint Sec 3.3 this is a design bug'})"
    )

    # ------------------------------------------------------------
    # Final side-by-side summary table
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL SUMMARY -- all blocks side by side")
    print("=" * 78)

    col_w = 14
    header = f"  {'metric':22s}" + "".join(f"{n:>{col_w}s}" for n in order)
    print(header)
    print("  " + "-" * (22 + col_w * len(order)))
    for key, label, passfn in METRIC_ORDER:
        row = f"  {label:22s}"
        for n in order:
            v = results[n][2][key]
            status = "P" if passfn(v) else "F"
            row += f"{fmt(v) + '(' + status + ')':>{col_w}s}"
        print(row)

    row = f"  {'T6 behav==0 exact':22s}"
    for n in order:
        v = results[n][2]["t6_behav_exact"]
        row += f"{str(v):>{col_w}s}"
    print(row)

    print(
        "\nNote: BASELINE column is the P0-only reference point used to compute "
        "Block A's delta; it is not one of the 3 official audited blocks (A/B/C)."
    )

    return results


if __name__ == "__main__":
    main()
