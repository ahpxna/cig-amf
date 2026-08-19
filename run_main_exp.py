import numpy as np
from run_experiment import make_main_env, make_runner, default_cfg

def run_comparison():
    cfg = default_cfg()
    
    # 1. Models participating in the race
    models_to_test = [
        "PureMeanField", 
        "FullExplicitLocal", 
        "Final-CIGAMF"
    ]
    
    # 2. Setting up the test: Behavioral Drift (Policy drift)
    # Increase episodes to 120 to allow models sufficient time to converge
    n_episodes = 120
    eval_every = 15
    n_seeds = 3
    
    results = {m: [] for m in models_to_test}

    for model in models_to_test:
        print(f"\n==========================================")
        print(f" ĐANG CHẠY MÔ HÌNH: {model}")
        print(f"==========================================")
        
        for seed in range(n_seeds):
            print(f"--- Seed {seed+1}/{n_seeds} ---")
            env = make_main_env(
                task_mode="behavioral_drift", 
                n_agents=24, 
                max_steps=30, 
                phase_length=40, 
                seed=seed * 100 + 42
            )
            
            runner = make_runner(model, env, cfg, device="cpu")
            history = runner.run(n_episodes=n_episodes, eval_every=eval_every)
            
            # Use the average reward of the last 20 episodes to evaluate convergence
            final_reward = np.mean(history["mean_reward"][-2:])
            results[model].append(final_reward)

    print("\n" + "="*60)
    print(" BẢNG KẾT QUẢ SO SÁNH CUỐI CÙNG (Mean Reward):")
    print("="*60)
    for model, rewards in results.items():
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f" -> {model:<20}: {mean_r:.4f} ± {std_r:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_comparison()
