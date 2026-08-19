import sys
import os

# 1. Search for modules within the 'envs' directory
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
    
    # --- COMPLETE ORIGINAL SOURCE CODE FROM TRAINING ---
    print("\n>>> RAW METRICS TỪ HÀM RUN_TIER0:")
    for key, val in out.items():
        if isinstance(val, float):
            print(f"  - {key:<25}: {val:.6f}")
        else:
            print(f"  - {key:<25}: {val}")
    
    # --- CREATE SIMILAR DECISION TABLE STEP 0 ---
    frac_changed = out.get("frac_action_changed", 0.0)
    norm_std = out.get("norm_by_std", 0.0)
    
    print("\n" + "="*65)
    print("KẾT QUẢ STEP 0 (PHIÊN BẢN TRUE ORACLE - ZERO COST):")
    print(f"Giá trị cấu trúc (frac_action_changed) : {frac_changed:.4f} (Kỳ vọng: > 0.05)")
    print(f"Độ phân tán ảnh hưởng (norm_by_std)    : {norm_std:.4f}")
    print("-" * 65)
    
    # [FIX-7] Old version phan "PASS ... Ready for CIG-AMF!" BASED ON
    # frac_action_changed > 0.05. That is a lower criterion than gate Experiment 0 of
    # paper: knows the structure has different actions not just distance
    # REGRET/reward is significant. Must satisfy BOTH (change action + high)
    # regret/reward is large enough to be considered passed gate.
    ok_changed = frac_changed > 0.05
    ok_regret = (norm_std == norm_std) and norm_std > 0.05   # NaN type
    if ok_changed and ok_regret:
        verdict = ("PASS: qua gate Experiment 0 — cấu trúc vừa đổi hành động "
                   f"({frac_changed:.1%}) vừa có hối tiếc chuẩn hoá đáng kể "
                   f"({norm_std:.3f}).")
    elif ok_changed and not ok_regret:
        verdict = ("CHƯA QUA GATE: hành động có đổi nhưng hối tiếc chuẩn hoá "
                   f"{norm_std:.3f} <= 0.05 — biết cấu trúc gần như vô ích về "
                   "mặt giá trị. KHÔNG được coi đây là 'sẵn sàng cho CIG-AMF'.")
    elif frac_changed > 0.0:
        verdict = ("CẢNH BÁO: cấu trúc có tác động nhưng quá yếu (< 5% số lần "
                   "đổi hành động).")
    else:
        verdict = "FAIL: Mù hay Oracle cũng ra hành động hệt nhau."
        
    print("Phán quyết (Verdict):", verdict)
    print("="*65)
