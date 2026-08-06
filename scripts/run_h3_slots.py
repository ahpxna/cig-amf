"""
H3 & RQ3 — Influence Signatures chống Slot Collapse + Adaptive Capacity (Eq 17).

LƯU Ý KỸ THUẬT: FinalCIGAMFRunner KHÔNG có method `evaluate()` và `run()` KHÔNG
trả về history (nó ghi vào runner.history). Nên script này chia run thành chunk
và đọc thẳng runner.history + get_slot_diagnostics() sau mỗi chunk, thay vì
monkey-patch `runner.evaluate` (sẽ không bao giờ được gọi -> mảng rỗng).

Output: results/h3/<variant>_seed<S>/eval.jsonl + summary.json
        results/h3/summary_h3.csv

Chạy:  python scripts/run_h3_slots.py --seeds 0 1 2 --episodes 200
"""
import argparse
import csv
import os

import numpy as np

from exp_common import (ROOT, append_jsonl, ensure_dir, last, make_args,
                        mean_over_egos, save_json)

import run_experiment as RE

ABLATIONS = [
    ("Full-CIGAMF",  {}),
    ("Scalar-Only",  {"proxy_effect_mode": "range"}),
    ("No-Semantic",  {"periph_use_uniform_mix": True, "periph_uniform_mix": 1.0}),
    ("No-AuxLoss",   {"periph_lb_coeff": 0.0}),
    ("Fixed-K",      {"belief_adaptive_k": False}),
]


def run_variant(name, cfg_over, seed, episodes, eval_every, device, out_root):
    out_dir = ensure_dir(os.path.join(out_root, f"{name}_seed{seed}"))
    jsonl = os.path.join(out_dir, "eval.jsonl")
    open(jsonl, "w").close()

    RE.set_global_seed(seed)
    cfg = RE.default_cfg()
    cfg["seed"] = seed
    cfg.update(cfg_over)
    args = make_args(seed=seed, device=device)

    env = RE.make_main_env(task_mode="behavioral_drift", n_agents=24,
                           max_steps=30, phase_length=40, seed=seed)
    runner = RE.make_runner("Final-CIGAMF", env, cfg, device)
    warmup = int(cfg.get("k0_warmup", 30))

    ents, cos_off, hitmax, ks, rewards = [], [], [], [], []
    n_chunks = max(1, episodes // eval_every)
    for c in range(n_chunks):
        runner.run(n_episodes=eval_every, eval_every=eval_every)
        ep = (c + 1) * eval_every
        hist = getattr(runner, "history", {})

        diag = {}
        try:
            diag = runner.periph_module.get_slot_diagnostics()
        except Exception:
            pass

        row = {
            "episode": ep,
            "post_warmup": int(ep > warmup),
            "usage_entropy_ratio": float(diag.get("usage_entropy_ratio", float("nan"))),
            "min_slot_usage": float(diag.get("min_usage", float("nan"))),
            "slot_cos_offdiag": float(diag.get("mean_offdiag_cosine",
                                    diag.get("offdiag_cosine", float("nan")))),
            "hit_max_rate": mean_over_egos(runner, "get_saturation_stats", "hit_max_rate"),
            "mean_core_size": last(hist, "core_size"),
            "mean_reward": last(hist, "mean_reward"),
            "f1": last(hist, "f1"),
            "throughput": last(hist, "throughput_agent_steps_per_sec"),
        }
        append_jsonl(jsonl, row)
        if row["post_warmup"]:          # chỉ tính SAU warm-up (bài học BUG N4)
            ents.append(row["usage_entropy_ratio"])
            cos_off.append(row["slot_cos_offdiag"])
            hitmax.append(row["hit_max_rate"])
            ks.append(row["mean_core_size"])
            rewards.append(row["mean_reward"])
        print(f"[H3 {name} s{seed}] ep={ep} entropy={row['usage_entropy_ratio']:.3f} "
              f"hit_max={row['hit_max_rate']:.3f} k={row['mean_core_size']:.2f}")

    def m(x):
        x = [v for v in x if np.isfinite(v)]
        return float(np.mean(x)) if x else float("nan")

    kmax = float(cfg.get("max_core_size", 4))
    summary = {
        "variant": name, "seed": seed, "episodes": episodes,
        "usage_entropy_ratio": m(ents),
        "slot_cos_offdiag": m(cos_off),
        "hit_max_rate": m(hitmax),
        "mean_core_size": m(ks),
        "frac_k_at_kmax": float(np.mean([1.0 for v in ks if np.isfinite(v)
                                         and v >= kmax - 1e-6]) if ks else 0.0),
        "mean_reward": m(rewards),
        "final_f1": last(getattr(runner, "history", {}), "f1"),
    }
    summary.update({f"cfg_{k}": v for k, v in cfg_over.items()})
    save_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu")
    a = ap.parse_args()

    out_root = ensure_dir(os.path.join(ROOT, "results", "h3"))
    rows = []
    for seed in a.seeds:
        for name, over in ABLATIONS:
            print(f"\n########## H3 {name} | seed {seed} ##########")
            rows.append(run_variant(name, over, seed, a.episodes,
                                    a.eval_every, a.device, out_root))

    keys = sorted({k for r in rows for k in r})
    p = os.path.join(out_root, "summary_h3.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[H3] saved {p}")
    print("[H3] Tiêu chí: Full entropy > 0.5 & cos_offdiag thấp; Scalar-Only/"
          "No-Semantic sập entropy HOẶC cos cao; Fixed-K có frac_k_at_kmax = 1.0.")


if __name__ == "__main__":
    main()
