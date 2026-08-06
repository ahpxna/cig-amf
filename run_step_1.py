import sys
import numpy as np
import torch
# Lấy từ file run_experiment.py ở thư mục gốc

from run_experiment import make_main_env, default_cfg, set_global_seed

# Import runner đúng từ thư mục runners/ (theo hình bạn chụp)
try:
    from runners.final_runner import FinalCIGAMFRunner
except ImportError:
    print("LỖI: Không tìm thấy FinalCIGAMFRunner trong runners/final_runner.py")
    sys.exit(1)

# Import hàm chẩn đoán từ models/ego_conditioned_latent.py
try:
    from models.ego_conditioned_latent import pair_specificity_score
except ImportError:
    print("LỖI: Không tìm thấy pair_specificity_score trong models/ego_conditioned_latent.py")
    sys.exit(1)

def run_step1():
    cfg = default_cfg()
    # [P-7 FINAL DEBUG] Bản cũ chạy 20 eps < k0_warmup=30 => TOÀN BỘ run nằm ở
    # Stage 0: belief kẹt prior (mu=0, sigma=1), proxy chưa train, và theo đúng
    # Eq (20)-(22) mọi neighbor bị dồn vào slot anomalous (sigma=1 > sigma_hi)
    # => Usage Entropy 0.15 là HÀNH VI ĐÚNG lúc warm-up, không phải slot collapse.
    # 4 chỉ số chẩn đoán chỉ có nghĩa khi đọc SAU khi sang Stage 1.
    cfg["k0_warmup"] = 10
    set_global_seed(cfg["seed"])
    print(f"[SEED-DEBUG] cfg_seed={cfg['seed']}")

    print("Đang khởi tạo môi trường và mô hình Final-CIGAMF...")
    env = make_main_env(
        task_mode="behavioral_drift",
        n_agents=24, 
        max_steps=30, 
        phase_length=40, 
        seed=42
    )
    
    runner = FinalCIGAMFRunner(env, cfg, device="cpu")
    
    print("Đang chạy 60 episodes (10 warm-up + 50 learned-stage)... (Vui lòng đợi ~5 phút)")
    runner.run(n_episodes=60, eval_every=10)
    
    print("\n" + "="*60)
    print(" KẾT QUẢ BƯỚC 1 - 4 CHỈ SỐ CHẨN ĐOÁN CỦA BẢN V2:")
    print("="*60)
    
    # 1. Ensemble Disagreement
    try:
        disagreement = runner.belief_modules[0].get_mean_uncertainty()
        print(f"[1] Ensemble Disagreement : {disagreement:.4f} \t(Kỳ vọng: > 0)")
    except Exception as e:
        print(f"[1] Ensemble Disagreement : [LỖI] {e}")

    # 2. Hit Max Rate
    try:
        hit_max_rate = runner.belief_modules[0].get_saturation_stats()["hit_max_rate"]
        print(f"[2] Hit Max Rate          : {hit_max_rate:.4f} \t(Kỳ vọng: < 1.0)")
    except Exception as e:
        print(f"[2] Hit Max Rate          : [LỖI] {e}")

    # 3. Usage Entropy Ratio
    try:
        entropy = runner.periph_module.get_slot_diagnostics()["usage_entropy_ratio"]
        print(f"[3] Usage Entropy Ratio   : {entropy:.4f} \t(Kỳ vọng: > 0.5)")
    except Exception as e:
        print(f"[3] Usage Entropy Ratio   : [LỖI] {e}")

    # 4. Specificity Ratio
    try:
        pair_rel = runner.pair_rel_module
        spec_ratio = pair_specificity_score(pair_rel, env.n_agents)["specificity_ratio"]
        print(f"[4] Specificity Ratio     : {spec_ratio:.4f} \t(Kỳ vọng: < 1.0)")
    except Exception as e:
        print(f"[4] Specificity Ratio     : [LỖI] {e}")

    print("="*60)

if __name__ == "__main__":
    run_step1()