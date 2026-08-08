import numpy as np
from models.diagnostics import structure_sensitivity_test
from run_experiment import make_main_env, make_runner, default_cfg

def run_one(condition, seed=0):
    """[FIX-6] Nhận seed từ structure_sensitivity_test để hai condition chạy
    trên CÙNG danh sách seed (paired). Bản cũ dùng
    seed=np.random.randint(0,1000) -> (a) không tái lập được, vi phạm nguyên
    tắc determinism 0.1 của debug doc; (b) hai nhánh chạy trên seed khác nhau
    nên chênh lệch reward lẫn cả phương sai giữa seed."""
    cfg = default_cfg()

    # Map tên condition sang tên mô hình thật.
    # Dùng FullExplicitLocal làm thế thân cho oracle_core
    # [FIX-O3] "oracle_core" giờ trỏ tới OracleCoreRunner THẬT (được cho sẵn
    # core đúng qua oracle |W*|, không tốn chi phí NHẬN DIỆN cấu trúc) — đây
    # mới là treatment mà Experiment 0 của paper yêu cầu. "explicit_local_learned"
    # giữ lại làm đối chứng baseline-vs-baseline, và tên của nó nói đúng bản
    # chất: một baseline PHẢI HỌC, không phải oracle.
    model_map = {
        "pure_mean_field": "PureMeanField",
        "oracle_core": "OracleCore",
        "random_core": "RandomCore",
        "explicit_local_learned": "FullExplicitLocal",
    }
    actual_model = model_map.get(condition, condition)

    # Khởi tạo môi trường
    env = make_main_env(
        task_mode="behavioral_drift",
        n_agents=24,
        max_steps=30,
        phase_length=40,
        seed=int(seed),
    )

    # Khởi tạo runner
    runner = make_runner(actual_model, env, cfg, device="cpu")

    # Cho chạy 60 episodes để mạng có thời gian học
    history = runner.run(n_episodes=60, eval_every=59)

    # Lấy mean_reward ở lần lưu cuối cùng
    final_reward = history["mean_reward"][-1]
    return final_reward

if __name__ == "__main__":
    print("Đang chạy Bước 0 - Kiểm tra độ nhạy cấu trúc môi trường...")
    # Truyền đúng 2 chữ mà diagnostics.py yêu cầu
    res = structure_sensitivity_test(
        run_fn=run_one,
        conditions=("pure_mean_field", "oracle_core"),
        n_seeds=8
    )

    print("\n" + "="*50)
    print("KẾT QUẢ STEP 0 (baseline-vs-baseline, KHÔNG phải Exp-0 oracle):")
    print(f"  baseline  : {res['baseline_condition']}")
    print(f"  treatment : {res['treatment_condition']}")
    print(f"  Structure Value : {res['structure_value']:.4f}")
    print(f"  CI95            : [{res['ci95'][0]:.4f}, {res['ci95'][1]:.4f}]"
          f"  significant={res['significant']}")
    print("Phán quyết (Verdict):", res["verdict"])
    print("\n[LƯU Ý] " + res["note"])
    print("Để lấy Exp-0 THẬT (oracle-core zero-cost), chạy: python run_oracle.py")
    print("="*50)
