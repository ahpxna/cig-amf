"""H1 & RQ1 — Identification & Calibration of Causality (epsilon-forcing + Doubly-Robust).

Run tiny-oracle calibration (run_experiment.run_tiny_task) on the configuration matrix:
  1. dr_eps003   : full DR, eps=0.03  (main)
  2. plugin      : disable DR correction  (isolate contribution of Eq 9)
  3. eps000/001/005 : sweep eps — eps=0 is observational control (paper Exp 1)

Output: results/h1/<variant>_seed<S>/tiny_oracle_summary.json (+ CSV per-pair)
        results/h1/summary_h1.csv

Default seeds: 0 1 2 3 4 5 6 7
Run:  python scripts/run_h1_calibration.py --seeds 0 1 2 3 4 5 6 7"""
import argparse
import contextlib
import csv
import json
import os

from exp_common import ROOT, ensure_dir, make_args, save_json  # noqa: F401

import run_experiment as RE

VARIANTS = [
    # (name, cfg override)
    ("dr_eps003", {"proxy_use_doubly_robust": True, "eps": 0.03}),
    ("plugin_eps003", {"proxy_use_doubly_robust": False, "eps": 0.03}),
    ("dr_eps000", {"proxy_use_doubly_robust": True, "eps": 0.0}),
    ("dr_eps001", {"proxy_use_doubly_robust": True, "eps": 0.01}),
    ("dr_eps005", {"proxy_use_doubly_robust": True, "eps": 0.05}),
    # [A1 spec] Only eps=0.05 separates from noise (Spearman +0.074, 99% CI
    # [+0.042,+0.109], 8/8 seed positive, survived Bonferroni). That is the first
    # evidence supporting H1 "bias decreases with eps" -> must sweep higher to get
    # THE SLOPE, which is the figure of Experiment 1.
    ("dr_eps008", {"proxy_use_doubly_robust": True, "eps": 0.08}),
    ("dr_eps012", {"proxy_use_doubly_robust": True, "eps": 0.12}),
]

KEEP_KEYS = (
    "rank_correlation_mean", "spearman_mean", "sign_agreement_mean",
    "signed_spearman_mean", "signed_mae_mean", "signed_bias_mean",
    "signed_rmse_mean", "signed_p_value_mean",
    "range_rank_correlation_mean",
    "bias_mean", "mae_mean", "pearson_mean", "n_states", "n_pairs",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7],
        help="Seed list for H1 calibration; default is 8 seeds 0..7",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-batch diagnostics; keep one concise line per run.",
    )
    ap.add_argument(
        "--summary_name", type=str, default="summary_h1.csv",
        help="Summary filename under results/h1 (useful for parallel shards).",
    )
    ap.add_argument(
        "--aggregate_only", action="store_true",
        help="Do not train; rebuild the requested summary from existing JSON files.",
    )
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
            print(f"\n########## H1 variant={name} seed={seed} ##########", flush=True)
            if not args_cli.aggregate_only:
                if args_cli.quiet:
                    with open(os.devnull, "w", encoding="utf-8") as sink:
                        with contextlib.redirect_stdout(sink):
                            RE.run_tiny_task(
                                args, cfg, args_cli.device, out_dir=out_dir,
                                run_label=f"h1_{name}_seed{seed}",
                            )
                else:
                    RE.run_tiny_task(
                        args, cfg, args_cli.device, out_dir=out_dir,
                        run_label=f"h1_{name}_seed{seed}",
                    )

            summ_path = os.path.join(out_dir, "tiny_oracle_summary.json")
            row = {"variant": name, "seed": seed}
            row.update(cfg_over)
            if os.path.exists(summ_path):
                with open(summ_path, encoding="utf-8") as f:
                    summ = json.load(f)
                for k, v in summ.items():
                    if k in KEEP_KEYS or isinstance(v, (int, float)):
                        row[k] = v
                print(
                    f"[H1] {name} seed={seed}: "
                    f"rank={float(summ.get('rank_correlation_mean', float('nan'))):+.4f} "
                    f"signed={float(summ.get('signed_spearman_mean', float('nan'))):+.4f} "
                    f"sign={float(summ.get('sign_agreement_mean', float('nan'))):.3f}",
                    flush=True,
                )
            else:
                print(f"[H1][WARN] missing {summ_path}", flush=True)
            rows.append(row)

    # summary table
    keys = sorted({k for r in rows for k in r})
    csv_path = os.path.join(out_root, os.path.basename(args_cli.summary_name))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[H1] saved {csv_path}")
    print("[H1] Tiêu chí đạt: rank-corr(dr_eps003) cao nhất, > plugin > dr_eps000;"
          " sign agreement >= 0.75.")


if __name__ == "__main__":
    main()
