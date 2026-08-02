import sys
import os
import numpy as np

# --- 1. XỬ LÝ ĐƯỜNG DẪN ---
# Lấy thư mục hiện tại (envs)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lùi lại 1 cấp để ra thư mục gốc (cig_amf)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Thêm thư mục gốc vào tầm nhìn của Python
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- 2. IMPORT CÁC MODULE TỪ ROOT ---
from models.diagnostics import bootstrap_ci, compare_two_methods
from runners.baseline_runner import PureMeanFieldRunner
from runners.final_runner import FinalCIGAMFRunner

# Môi trường nằm cùng thư mục envs
try:
    from omni_arena import OmniArena
except ModuleNotFoundError:
    from envs.omni_arena import OmniArena

# --- 3. HÀM KẾT NỐI (TRAIN & EVAL) ---
def train_and_eval(config, seed, n_episodes, hidden, lr, batch_size, k_core):
    print(f"\n>>> ĐANG CHẠY CẤU HÌNH: {config} | SEED: {seed} <<<")
    
    env = OmniArena(
        enable_conditional_gates=True, 
        enable_latency_ladder=True,
        enable_congestion=True,
        enable_structural_shift=True,
        seed=seed
    )
    
    cfg = {
        "policy_hidden": hidden,
        "policy_lr": lr,
        "proxy_batch_size": batch_size,
        "seed": seed,
        "discount": 0.95,
        "core_dim": 64,       
        "periph_dim": 32,     
        "belief_dim": 32,
        
        # Các thông số bắt buộc của FinalCIGAMFRunner
        "n_ensemble": 4,
        "proxy_lr": 1e-3,
        "core_lr": 5e-4,
        "shadow_dim": 16,
        "num_memory_slots": 4,
        "periph_memory_dim": 32,
        "belief_top_k": 3,
        "belief_pooled_hidden": 64,
        "k0_warmup": 10,
        "causal_horizon": 8,
        "belief_lambda_0": 0.12,
        "belief_uncertainty_scale": 2.0,
        "belief_tau": 0.10,
        "belief_tau_in": 0.62,
        "belief_tau_out": 0.46,
        "seed_core_top_k": k_core,
        "proxy_train_steps": 5,
        "proxy_holdout_size": 256,
        "bc_train_steps": 5,
        "bc_batch_size": batch_size,
    }

    if config == "pure_mf":
        runner = PureMeanFieldRunner(env, cfg)
    elif config == "cig_amf":
        runner = FinalCIGAMFRunner(env, cfg)
    elif config in ["random_core", "oracle_core"]:
        print(f"Warning: Chưa implement {config} cụ thể, tạm dùng FinalCIGAMFRunner")
        runner = FinalCIGAMFRunner(env, cfg)
    else:
        raise ValueError(f"Cấu hình không hợp lệ: {config}")

    history = runner.run(n_episodes=n_episodes, eval_every=1)
    
    final_reward = np.mean(history["mean_reward"][-5:]) if len(history["mean_reward"]) >= 5 else history["mean_reward"][-1]
    
    return {"final_window_reward": final_reward}


# --- 4. KỊCH BẢN CHẠY CHÍNH ---
if __name__ == "__main__":
    # Đang set nhỏ để chạy test thử, sau khi thành công bạn đổi thành N_EP = 100, SEEDS = list(range(8))
    N_EP = 100
    CONFIGS = ["random_core", "pure_mf", "cig_amf", "oracle_core"]
    SEEDS = list(range(8))

    results = {c: [] for c in CONFIGS}

    for cfg in CONFIGS:
        for seed in SEEDS:
            r = train_and_eval(
                config=cfg, seed=seed,
                n_episodes=N_EP,
                hidden=160, lr=3e-4, batch_size=256, k_core=3,
            )
            results[cfg].append(r["final_window_reward"])

    R = {c: np.mean(results[c]) for c in CONFIGS}

    structure_value = R["oracle_core"] - R["pure_mf"]
    learning_range  = R["oracle_core"] - R["random_core"]

    sv_norm = structure_value / max(1e-12, learning_range)
    rho     = (R["cig_amf"] - R["pure_mf"]) / max(1e-12, structure_value)

    print(f"\nstructure_value (thô)      = {structure_value:.4f}")
    print(f"structure_value (chuẩn hoá)= {sv_norm:.3f}")
    print(f"rho (phần bắt được)        = {rho:.3f}")

    print("\nKiểm định Oracle vs Pure MF:")
    print(compare_two_methods(results["oracle_core"], results["pure_mf"]))
    print("\nKiểm định CIG-AMF vs Pure MF:")
    print(compare_two_methods(results["cig_amf"],     results["pure_mf"]))