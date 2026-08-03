"""
H1 & RQ1 — Nhận diện & Hiệu chuẩn nhân quả (epsilon-forcing + Doubly-Robust).

Chạy tiny-oracle calibration (run_experiment.run_tiny_task) trên ma trận cấu hình:
  1. dr_eps003   : full DR, eps=0.03  (chính)
  2. plugin      : tắt DR correction  (cô lập đóng góp của Eq 9)
  3. eps000/001/005 : quét eps — eps=0 là observational control (paper Exp 1)

Output: results/h1/<variant>_seed<S>/tiny_oracle_summary.json (+ CSV per-pair)
        results/h1/summary_h1.csv

Chạy:  python scripts/run_h1_calibration.py --seeds 0 1 2
"""
import argparse
import csv
import json
import os

from exp_common import ROOT, ensure_dir, make_args, save_json  # noqa: F401

import run_experiment as RE

VARIANTS = [
    # (tên, cfg override)
    ("dr_eps003", {"proxy_use_doubly_robust": True, "eps": 0.03}),
    ("plugin_eps003", {"proxy_use_doubly_robust": False, "eps": 0.03}),
    ("dr_eps000", {"proxy_use_doubly_robust": True, "eps": 0.0}),
    ("dr_eps001", {"proxy_use_doubly_robust": True, "eps": 0.01}),
    ("dr_eps005", {"proxy_use_doubly_robust": True, "eps": 0.05}),
]

KEEP_KEYS = (
    "rank_correlation_mean", "spearman_mean", "sign_agreement_mean",
    "bias_mean", "mae_mean", "pearson_mean", "n_states", "n_pairs",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cpu")
    args_cli = ap.parse_args()

    out_root = ensure_dir(os.path.join(ROOT, "results", "h1"))
    rows = []

    for seed in args_cli.seeds:
        for name, cfg_over in VARIANTS:
            out_dir = ensure_dir(os.path.join(out_root, f"{name}_seed{seed}"))
            cfg = RE.default_cfg()
            cfg.update(cfg_over)
            args = make_args(seed=seed, device=args_cli.device,
                             result_dir=out_dir)
            print(f"\n########## H1 variant={name} seed={seed} ##########")
            RE.run_tiny_task(args, cfg, args_cli.device, out_dir=out_dir,
                             run_label=f"h1_{name}_seed{seed}")

            summ_path = os.path.join(out_dir, "tiny_oracle_summary.json")
            row = {"variant": name, "seed": seed}
            row.update(cfg_over)
            if os.path.exists(summ_path):
                with open(summ_path, encoding="utf-8") as f:
                    summ = json.load(f)
                for k, v in summ.items():
                    if k in KEEP_KEYS or isinstance(v, (int, float)):
                        row[k] = v
            rows.append(row)

    # bảng tổng hợp
    keys = sorted({k for r in rows for k in r})
    csv_path = os.path.join(out_root, "summary_h1.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[H1] saved {csv_path}")
    print("[H1] Tiêu chí đạt: rank-corr(dr_eps003) cao nhất, > plugin > dr_eps000;"
          " sign agreement >= 0.75.")


if __name__ == "__main__":
    main()
