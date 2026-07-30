import numpy as np
from models.diagnostics import structure_sensitivity_test
from run_experiment import make_main_env, make_runner, default_cfg

def run_one(condition):
    cfg = default_cfg()

    # Map tên condition sang tên mô hình thật.
    # Dùng FullExplicitLocal làm thế thân cho oracle_core
    model_map = {
        "pure_mean_field": "PureMeanField",
        "oracle_core": "FullExplicitLocal"
    }
    actual_model = model_map.get(condition, condition)

    # Khởi tạo môi trường
    env = make_main_env(
        task_mode="behavioral_drift",
        n_agents=24,
        max_steps=30,
        phase_length=40,
        seed=np.random.randint(0, 1000)
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
        n_seeds=3
    )

    print("\n" + "="*50)
    print("KẾT QUẢ STEP 0:")
    print("Giá trị cấu trúc (Structure Value):", res["structure_value"])
    print("Phán quyết (Verdict):", res["verdict"])
    print("="*50)
