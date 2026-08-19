import numpy as np
from models.diagnostics import structure_sensitivity_test
from run_experiment import make_main_env, make_runner, default_cfg

def run_one(condition, seed=0):
    """Run paired conditions on the seed supplied by the sensitivity test.

    [FIX-6] The previous implementation used
    ``seed=np.random.randint(0, 1000)``. That was non-reproducible, violated
    debugging determinism rule 0.1, and evaluated the two branches on different
    seeds, mixing between-seed variance into the reward difference.
    """
    cfg = default_cfg()

    # Map condition labels to concrete runner names.
    # [FIX-O3] ``oracle_core`` now selects the real OracleCoreRunner, which is
    # given the correct core through oracle |W*| at zero structure-identification
    # cost. This is the treatment required by paper Experiment 0.
    # ``explicit_local_learned`` remains as a baseline-vs-baseline control; its
    # name reflects that it must learn and is not an oracle.
    model_map = {
        "pure_mean_field": "PureMeanField",
        "oracle_core": "OracleCore",
        "random_core": "RandomCore",
        "explicit_local_learned": "FullExplicitLocal",
    }
    actual_model = model_map.get(condition, condition)

    # Initialize the environment.
    env = make_main_env(
        task_mode="behavioral_drift",
        n_agents=24,
        max_steps=30,
        phase_length=40,
        seed=int(seed),
    )

    # Initialize the runner.
    runner = make_runner(actual_model, env, cfg, device="cpu")

    # Run 60 episodes to allow the network to learn.
    history = runner.run(n_episodes=60, eval_every=59)

    # Return mean reward from the final recorded evaluation.
    final_reward = history["mean_reward"][-1]
    return final_reward

if __name__ == "__main__":
    print("Đang chạy Bước 0 - Kiểm tra độ nhạy cấu trúc môi trường...")
    # Pass exactly the two condition labels required by diagnostics.py.
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
