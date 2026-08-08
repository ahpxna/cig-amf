"""
H2 & RQ2 — Sự tách biệt tầng & Tính chọn lọc (Selectivity Ratio, Eq 33).

Với mỗi model x seed, chạy 2 run cùng cấu hình:
  - behavioral_drift  -> Delta_behav  (kỳ vọng NHỎ: dependencies cố định)
  - structural_shift  -> Delta_struct (kỳ vọng LỚN tại/ngay sau shift)
SR = mean(Delta_struct quanh shift) / mean(Delta_behav).
Paper: CIG-AMF cho SR >> 1; baseline tương quan/attention cho SR ~ 1.

Ngoài SR còn log: core F1, recovery latency (đã TRỪ độ trễ H bước của trigger),
trigger event times (CUSUM Eq 32), reward trajectory.

Output: results/h2/<model>_<mode>_seed<S>/eval.jsonl + summary.json
        results/h2/summary_h2.csv

Chạy:  python scripts/run_h2_selectivity.py --seeds 0 1 2 --episodes 400
"""
import argparse
import csv
import os

import numpy as np

from exp_common import (ROOT, append_jsonl, delta_norm, ensure_dir, last,
                        make_args, save_json, w_matrix)

import run_experiment as RE

MODELS = ["Final-CIGAMF", "NoTwoTimescale", "PureMeanField"]
MODES = ["behavioral_drift", "structural_shift"]


def run_one(model, mode, seed, episodes, eval_every, device, out_root):
    out_dir = ensure_dir(os.path.join(out_root, f"{model}_{mode}_seed{seed}"))
    jsonl = os.path.join(out_dir, "eval.jsonl")
    open(jsonl, "w").close()

    RE.set_global_seed(seed)
    cfg = RE.default_cfg()
    cfg["seed"] = seed
    args = make_args(seed=seed, device=device)

    env = RE.make_main_env(task_mode=mode, n_agents=24, max_steps=30,
                           phase_length=40, seed=seed)
    runner = RE.make_runner(model, env, cfg, device)

    H = int(cfg.get("causal_horizon", 8))
    has_belief = hasattr(runner, "belief_modules")
    W_prev = w_matrix(runner, env.n_agents) if has_belief else None
    deltas, shift_eps, trigger_eps = [], [], []

    # [FIX-H2a] Phát hiện structural shift bằng CHÍNH bảng ảnh hưởng ground
    # truth, không phải active_lane. active_lane bị reset()/restore_state ghi
    # đè giữa các chunk nên so sánh nó gần như luôn cho False => shift_eps
    # rỗng => delta_struct = mean(mảng rỗng) = NaN => SR = NaN. Đó là lý do
    # TOÀN BỘ summary_h2.csv trước đây là NaN.
    def _phi_fingerprint(e):
        try:
            return tuple(
                (int(ego), tuple(sorted((int(k), round(float(v), 6))
                                        for k, v in row.items())))
                for ego, row in sorted(e.gt_influence_by_ego.items())
            )
        except Exception:
            return None

    prev_phi = _phi_fingerprint(env)

    n_chunks = max(1, episodes // eval_every)
    for c in range(n_chunks):
        runner.run(n_episodes=eval_every, eval_every=eval_every)
        ep = (c + 1) * eval_every
        hist = getattr(runner, "history", {})

        d = float("nan")
        if has_belief:
            W_now = w_matrix(runner, env.n_agents)
            d = delta_norm(W_prev, W_now)
            W_prev = W_now

        phi_now = _phi_fingerprint(env)
        is_shift = (prev_phi is not None and phi_now is not None
                    and phi_now != prev_phi)
        prev_phi = phi_now
        if is_shift:
            shift_eps.append(ep)

        trig = int(last(hist, "triggered", 0) or 0)
        if trig:
            trigger_eps.append(ep)

        row = {
            "episode": ep, "delta": d, "is_shift_window": int(is_shift),
            "triggered": trig,
            # [FIX-3] Key ĐÚNG trong runner.history là mean_f1 / mean_reward /
            # mean_core_size. Dùng "f1"/"reward"/"core_size" -> luôn NaN, đây
            # chính là nhóm cột NaN đang treo trong summary H2/H3.
            "f1": last(hist, "mean_f1"), "reward": last(hist, "mean_reward"),
            "core_size": last(hist, "mean_core_size"),
            "tier_separation_ratio": float(
                getattr(env, "tier_separation_ratio", lambda: float("nan"))()),
            # [C4] CSD-proxy: số cặp đang coupling ở cuối chunk.
            "n_close_coupling_pairs": int(
                env.close_coupling_pairs()
                if hasattr(env, "close_coupling_pairs") else -1
            ),
        }
        append_jsonl(jsonl, row)
        deltas.append((ep, d, is_shift))
        print(f"[H2 {model}/{mode} s{seed}] ep={ep} delta={d:.4f} "
              f"f1={row['f1']:.3f} trig={trig}")

    # --- SR + recovery latency ---
    finite_rows = [(ep, d) for ep, d, _ in deltas if np.isfinite(d)]
    dvals = np.array([d for _, d in finite_rows], dtype=np.float64)
    struct_mask = np.array(
        [any(0 <= ep - se <= 2 * eval_every for se in shift_eps)
         for ep, _ in finite_rows],
        dtype=bool,
    )

    d_struct = float(np.mean(dvals[struct_mask])) if struct_mask.any() else float("nan")
    d_behav = float(np.mean(dvals[~struct_mask])) if (~struct_mask).any() else float("nan")
    # [FIX-H2b] Chia cho (d_behav + 1e-12) làm SR nổ khi d_behav ~ 0 — đúng
    # lớp lỗi chia-gần-0 đã gặp ở T5/T6/structure_value. Yêu cầu mẫu số phải
    # có độ lớn tối thiểu mới cho ra tỉ số.
    if (np.isfinite(d_struct) and np.isfinite(d_behav) and d_behav > 1e-6):
        sr = float(d_struct / d_behav)
    else:
        sr = float("nan")

    # recovery latency: số eval-interval từ shift tới khi f1 hồi >= 90% mức tiền-shift,
    # trừ đi độ trễ H bước của trigger (paper yêu cầu account cho H).
    f1s = [r for r in deltas]  # placeholder giữ thứ tự; đọc f1 từ file
    lat = float("nan")
    if shift_eps:
        import json as _json
        recs = [_json.loads(l) for l in open(jsonl, encoding="utf-8")]
        se = shift_eps[0]
        pre = [r["f1"] for r in recs if r["episode"] < se and np.isfinite(r["f1"])]
        if pre:
            target = 0.9 * float(np.mean(pre[-3:]))
            for r in recs:
                if r["episode"] >= se and np.isfinite(r["f1"]) and r["f1"] >= target:
                    lat = (r["episode"] - se) / eval_every - H / max(1, eval_every)
                    break

    summary = {
        "model": model, "mode": mode, "seed": seed, "episodes": episodes,
        "delta_mean_struct": d_struct, "delta_mean_behav": d_behav,
        "selectivity_ratio": sr,
        "n_shift_events": len(shift_eps), "shift_episodes": shift_eps,
        "n_triggers": len(trigger_eps), "trigger_episodes": trigger_eps,
        "recovery_latency_intervals": lat,
        "final_f1": last(getattr(runner, "history", {}), "mean_f1"),
        "final_reward": last(getattr(runner, "history", {}), "mean_reward"),
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--models", type=str, nargs="+", default=MODELS)
    ap.add_argument("--device", type=str, default="cpu")
    a = ap.parse_args()

    out_root = ensure_dir(os.path.join(ROOT, "results", "h2"))
    rows = []
    for seed in a.seeds:
        for model in a.models:
            per_mode = {}
            for mode in MODES:
                per_mode[mode] = run_one(model, mode, seed, a.episodes,
                                         a.eval_every, a.device, out_root)
            # SR liên-run: struct từ run structural, behav từ run drift
            ds = per_mode["structural_shift"]["delta_mean_struct"]
            db = per_mode["behavioral_drift"]["delta_mean_behav"]
            sr_cross = (
                float(ds / db)
                if (np.isfinite(ds) and np.isfinite(db) and db > 1e-6)
                else float("nan")
            )
            rows.append({
                "model": model, "seed": seed,
                "delta_struct": ds, "delta_behav": db,
                "SR_cross_run": sr_cross,
                "SR_within_structural": per_mode["structural_shift"]["selectivity_ratio"],
                "recovery_latency": per_mode["structural_shift"]["recovery_latency_intervals"],
                "n_triggers": per_mode["structural_shift"]["n_triggers"],
                "final_f1_struct": per_mode["structural_shift"]["final_f1"],
            })

    keys = sorted({k for r in rows for k in r})
    p = os.path.join(out_root, "summary_h2.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[H2] saved {p}")
    print("[H2] Tiêu chí đạt: SR(Final-CIGAMF) >> 1, SR(baseline) ~ 1; "
          "F1 hồi phục sau shift với latency hữu hạn + có trigger tương ứng.")


if __name__ == "__main__":
    main()
