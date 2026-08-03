"""
Gom H1/H2/H3 -> bảng tổng hợp (mean +/- std qua seed) cho phần Discussion.
Không phụ thuộc pandas: dùng csv + statistics thuần.

Chạy:  python scripts/collect_results.py
Ghi:   results/summary_tables.md  (+ in ra màn hình)
"""
import csv
import os
import statistics as st

from exp_common import ROOT

SPECS = [
    ("H1 — RQ1: Nhận diện & Hiệu chuẩn nhân quả",
     os.path.join(ROOT, "results", "h1", "summary_h1.csv"), "variant",
     ["rank_correlation_mean", "spearman_mean", "sign_agreement_mean",
      "bias_mean", "mae_mean"]),
    ("H2 — RQ2: Tách tầng & Tính chọn lọc (Eq 33)",
     os.path.join(ROOT, "results", "h2", "summary_h2.csv"), "model",
     ["delta_behav", "delta_struct", "SR_cross_run", "recovery_latency",
      "n_triggers", "final_f1_struct"]),
    ("H3 — RQ3: Slot collapse & Capacity allocation",
     os.path.join(ROOT, "results", "h3", "summary_h3.csv"), "variant",
     ["usage_entropy_ratio", "slot_cos_offdiag", "hit_max_rate",
      "mean_core_size", "frac_k_at_kmax", "mean_reward"]),
]


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None       # loại NaN
    except (TypeError, ValueError):
        return None


def table(path, group_key, metrics):
    if not os.path.exists(path):
        return f"_(chưa có {os.path.relpath(path, ROOT)} — chạy script tương ứng trước)_\n"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "_(file rỗng)_\n"

    cols = [m for m in metrics if m in rows[0]]
    groups = {}
    for r in rows:
        groups.setdefault(r.get(group_key, "?"), []).append(r)

    out = ["| " + group_key + " | n | " + " | ".join(cols) + " |",
           "|" + "---|" * (len(cols) + 2)]
    for g, rs in groups.items():
        cells = []
        for c in cols:
            vals = [v for v in (fnum(r.get(c)) for r in rs) if v is not None]
            if not vals:
                cells.append("—")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.3f}")
            else:
                cells.append(f"{st.mean(vals):.3f} ± {st.pstdev(vals):.3f}")
        out.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    md = ["# CIG-AMF — Bảng kết quả tổng hợp (mean ± std qua seed)\n"]
    for title, path, key, metrics in SPECS:
        md.append(f"\n## {title}\n")
        md.append(table(path, key, metrics))

    md.append("\n## Đọc kết quả\n")
    md.append("- **H1 đạt** khi `dr_eps003` có rank-corr cao nhất, "
              "> `plugin_eps003` > `dr_eps000`; sign agreement ≥ 0.75.\n")
    md.append("- **H2 đạt** khi `SR_cross_run`(Final-CIGAMF) ≫ 1 và "
              "≈ 1 với PureMeanField; recovery latency hữu hạn + có trigger.\n")
    md.append("- **H3 đạt** khi `Full-CIGAMF` giữ entropy > 0.5 với "
              "`slot_cos_offdiag` thấp, còn `Scalar-Only`/`No-Semantic` sập; "
              "`Fixed-K` có `frac_k_at_kmax` = 1.0 (allocation là hằng số).\n")

    text = "".join(x if x.endswith("\n") else x + "\n" for x in md)
    out = os.path.join(ROOT, "results", "summary_tables.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"[collect] saved {out}")


if __name__ == "__main__":
    main()
