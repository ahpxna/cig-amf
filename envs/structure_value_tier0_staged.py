"""Run cumulative Tier 0 oracle regret across five P0-to-P4 stages.

The staged comparison identifies which environment component changes the
value of structural information.
"""

import numpy as np

from omni_arena import OmniArena
from structure_value_tier0 import run_tier0

STAGES = [
    (
        "1. BASELINE (Tất cả TẮT)", 
        dict(enable_conditional_gates=False, enable_latency_ladder=False, enable_congestion=False, enable_structural_shift=False)
    ),
    (
        "2. +P1 (Conditional gates)", 
        dict(enable_conditional_gates=True, enable_latency_ladder=False, enable_congestion=False, enable_structural_shift=False)
    ),
    (
        "3. +P2 (Latency ladder)", 
        dict(enable_conditional_gates=True, enable_latency_ladder=True, enable_congestion=False, enable_structural_shift=False)
    ),
    (
        "4. +P3 (Congestion - Rủi ro nuốt tín hiệu)", 
        dict(enable_conditional_gates=True, enable_latency_ladder=True, enable_congestion=True, enable_structural_shift=False)
    ),
    (
        "5. +P4 (Structural shift - FULL)",
        # Task #13 root cause for P3 and P4 matching to every digit:
        # enable_structural_shift=True alone does not cause a shift.
        # _maybe_structural_shift() returns when mode != "structural_shift";
        # OmniArena defaults to mode="behavioral_drift". This check occurs
        # before the enable flag or phase boundary. Without an explicit mode,
        # _do_structural_shift() never runs, Phi never changes, and stages 4
        # and 5 execute the same path with seed=123. Their exact equality is
        # deterministic, not coincidence or a missed trigger window.
        #
        # env_audit_staged.py already follows the correct T6 pattern: it uses
        # a separate structural probe with mode="structural_shift", while the
        # main T1-T5 environment intentionally remains in behavioral_drift.
        #
        # The fix sets structural_shift mode only for this stage. Stages 1-4
        # have enable_structural_shift=False, and their flag guard returns
        # before mode is read. phase_length is also reduced from 40 to 2 so a
        # real shift occurs inside the short staged sample. With N_STATES=10,
        # steps_between=8, and max_steps=60, the guard
        # ``t+horizon+steps_between >= max_steps`` in run_tier0() causes one
        # reset around state 6/10. episode_count changes from 1 (the reset in
        # __init__) to 2. At phase_length=40, 2%40 != 0 and no shift occurs;
        # at phase_length=2, 2%2 == 0 and states 6-9 observe post-shift Phi.
        dict(enable_conditional_gates=True, enable_latency_ladder=True, enable_congestion=True, enable_structural_shift=True,
             mode="structural_shift", phase_length=2)
    ),
]

if __name__ == "__main__":
    print("Bắt đầu quét TẦNG 0 theo 5 giai đoạn tích luỹ...")
    results = {}
    
    # Use 10 states so all five stages complete in roughly 15-20 minutes.
    N_STATES = 10 
    
    for name, flags in STAGES:
        print("\n" + "="*70)
        print(f"ĐANG CHẠY: {name}")
        print(f"Flags: {flags}")
        print("="*70)
        
        env = OmniArena(
            n_agents=24, grid_size=24, n_zones=4,
            **flags
        )
        
        out, _ = run_tier0(
            env=env, 
            n_states=N_STATES, 
            steps_between=8, 
            k_core=3, 
            horizon=8, 
            seed=123
        )
        results[name] = out
        
    print("\n" + "="*100)
    print("BẢNG TỔNG HỢP TẦNG 0 THEO GIAI ĐOẠN (HỐI TIẾC ORACLE)")
    print("="*100)
    # Task #12: q_range-based normalised_regret is contaminated by the WORST
    # action sequence, which is never selected. A larger penalty can inflate
    # q_range without increasing frac_action_changed. norm_by_std, using
    # std(Q) as denominator, is the primary normalized metric.
    # q_range-based normalised_regret is retained only for backward comparison.
    header = (
        f"{'Giai đoạn':<40} | {'Tỷ lệ đổi HĐ':<13} | {'ChuẩnHoá(range)':<16} "
        f"| {'ChuẩnHoá(std)':<14} | {'HT|đã đổi':<11} | {'P95 HT'}"
    )
    print(header)
    print("-" * 100)

    def _fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NaN"
        return f"{x:.3f}"

    for name, _ in STAGES:
        out = results[name]
        changed = f"{out['frac_action_changed']:.1%}"
        regret_range = f"{out['normalised_regret']:.3f}"
        regret_std = _fmt(out.get("norm_by_std"))
        regret_given_changed = _fmt(out.get("regret_given_changed"))
        p95 = f"{out['p95_regret']:.3f}" if "p95_regret" in out else "n/a"
        print(
            f"{name:<40} | {changed:>13} | {regret_range:>16} "
            f"| {regret_std:>14} | {regret_given_changed:>11} | {p95:>7}"
        )

    print("-" * 100)
    print(">> TIÊU CHÍ: Tỷ lệ đổi hành động > 10% VÀ Hối tiếc chuẩn hoá(std) > 0.05 là PASS.")
    print(">> (Hối tiếc chuẩn hoá(range) giữ lại để so sánh ngược -- xem LƯU Ý trong")
    print(">>  structure_value_tier0.py về việc mẫu số của nó bị nhiễu bởi chuỗi tệ nhất.)")
