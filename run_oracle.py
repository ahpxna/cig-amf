import sys
import os

# 1. Ép Python tìm kiếm module bên trong thư mục 'envs'
current_dir = os.path.dirname(os.path.abspath(__file__))
envs_dir = os.path.join(current_dir, 'envs')
if envs_dir not in sys.path:
    sys.path.append(envs_dir)

from omni_arena import OmniArena
from structure_value_tier0 import run_tier0

if __name__ == "__main__":
    print("================================================================")
    print("ĐANG CHẠY TRUE ORACLE-CORE (ZERO-COST) ĐỂ ĐO STRUCTURE VALUE THẬT")
    print("================================================================")
    
    env = OmniArena(
        n_agents=24, 
        grid_size=24, 
        n_zones=4,
        enable_conditional_gates=True, 
        enable_latency_ladder=True,
        enable_congestion=True, 
        enable_structural_shift=True
    )
    
    out, rows = run_tier0(env=env, n_states=20, steps_between=10, k_core=3, horizon=8, seed=123)
    
    # --- IN TOÀN BỘ CHỈ SỐ GỐC TỪ TIỀN TRÌNH ---
    print("\n>>> RAW METRICS TỪ HÀM RUN_TIER0:")
    for key, val in out.items():
        if isinstance(val, float):
            print(f"  - {key:<25}: {val:.6f}")
        else:
            print(f"  - {key:<25}: {val}")
    
    # --- TỰ TẠO BẢNG PHÁN QUYẾT GIỐNG STEP 0 ---
    frac_changed = out.get("frac_action_changed", 0.0)
    norm_std = out.get("norm_by_std", 0.0)
    
    print("\n" + "="*65)
    print("KẾT QUẢ STEP 0 (PHIÊN BẢN TRUE ORACLE - ZERO COST):")
    print(f"Giá trị cấu trúc (frac_action_changed) : {frac_changed:.4f} (Kỳ vọng: > 0.05)")
    print(f"Độ phân tán ảnh hưởng (norm_by_std)    : {norm_std:.4f}")
    print("-" * 65)
    
    if frac_changed > 0.05:
        verdict = "PASS: Môi trường TỐT. Trí tuệ Oracle thay đổi hẳn hành động so với Mù (> 5%). Đã sẵn sàng cho CIG-AMF!"
    elif frac_changed > 0.0:
        verdict = "CẢNH BÁO NHẸ: Cấu trúc có tác động, nhưng quá yếu (< 5%). Hãy tăng hình phạt va chạm trong envs."
    else:
        verdict = "FAIL: Mù hay Oracle cũng ra hành động hệt nhau. Structure Value = 0. Cần sửa môi trường."
        
    print("Phán quyết (Verdict):", verdict)
    print("="*65)