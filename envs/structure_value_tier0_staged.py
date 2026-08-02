"""
Chạy Tầng 0 (Hối tiếc Oracle) tích luỹ qua 5 giai đoạn P0 -> P4
để chẩn đoán xem thành phần nào của môi trường làm ảnh hưởng đến giá trị của cấu trúc.
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
        # LƯU Ý (Task #13 -- root cause của bug "P3 và P4 giống hệt nhau tới
        # từng chữ số"): enable_structural_shift=True KHÔNG đủ để shift thật
        # sự xảy ra. _maybe_structural_shift() trong omni_arena.py trả về
        # sớm ngay khi self.mode != "structural_shift" (mặc định của
        # OmniArena.__init__ là mode="behavioral_drift") -- TRƯỚC CẢ khi
        # kiểm tra cờ enable_structural_shift hay ranh giới phase. Vì stage
        # này (giống mọi stage khác) không truyền mode=, env luôn chạy
        # "behavioral_drift" -> _do_structural_shift() không bao giờ được
        # gọi -> Phi không đổi -> stage 4 và stage 5 chạy CÙNG một đường mã,
        # CÙNG seed=123 -> kết quả giống hệt nhau tới từng chữ số một cách
        # TẤT ĐỊNH (không phải trùng hợp, không phải shift "không kịp
        # trigger trong cửa sổ mẫu"). Đây đúng là pattern env_audit_staged.py
        # đã dùng cho T6 (probe structural riêng với mode="structural_shift",
        # xem run_t6() trong env_audit.py) -- main env T1-T5 ở đó CỐ Ý luôn
        # giữ "behavioral_drift" và không bao giờ tự shift.
        # Sửa: truyền mode="structural_shift" CHỈ cho stage này (không đụng
        # stage 1-4, vốn có enable_structural_shift=False nên mode không có
        # tác dụng gì với chúng -- _maybe_structural_shift() trả về ở dòng
        # kiểm tra cờ TRƯỚC CẢ khi đọc self.mode). Đồng thời rút ngắn
        # phase_length xuống 2 (mặc định 40) để đảm bảo một shift THỰC SỰ
        # trigger trong cửa sổ mẫu ngắn của staged run: với N_STATES=10,
        # steps_between=8, max_steps=60 mặc định, guard "t+horizon+
        # steps_between >= max_steps" trong run_tier0() chỉ kích hoạt đúng
        # MỘT lần reset() trong suốt vòng lặp (khoảng state thứ 6/10) --
        # episode_count đi từ 1 (do __init__ gọi reset() một lần) lên 2 tại
        # lần reset() đó. Với phase_length=40 mặc định, 2 % 40 != 0 -> không
        # bao giờ shift dù có mode đúng. Với phase_length=2, 2 % 2 == 0 ->
        # shift trigger đúng tại lần reset() đó, cho nửa sau của mẫu (state
        # 6-9) phản ánh cấu trúc Phi SAU shift.
        dict(enable_conditional_gates=True, enable_latency_ladder=True, enable_congestion=True, enable_structural_shift=True,
             mode="structural_shift", phase_length=2)
    ),
]

if __name__ == "__main__":
    print("Bắt đầu quét TẦNG 0 theo 5 giai đoạn tích luỹ...")
    results = {}
    
    # Rút ngắn n_states xuống 10 để chạy 5 lượt không quá lâu (tổng ~15-20 phút)
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
    # Task #12: q_range-based normalised_regret bị nhiễu bởi chuỗi hành động
    # TỆ NHẤT (không bao giờ được chọn) -- tăng penalty có thể phình to
    # q_range giả tạo mà không hề tăng frac_action_changed. norm_by_std
    # (std(Q) làm mẫu số) là chỉ số chuẩn hoá CHÍNH; giữ normalised_regret
    # (q_range) lại chỉ để so sánh ngược.
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