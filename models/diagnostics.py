"""
diagnostics.py — Ba thí nghiệm chẩn đoán mà paper ĐANG THIẾU.

=============================================================================
[D1] SELECTIVE RESPONSIVENESS — hình headline cho paper
=============================================================================
Đóng góp KHÁI NIỆM MẠNH NHẤT của paper là tách non-stationarity thành hai
tầng (structural: AI ảnh hưởng ai / behavioural: họ đang hành xử thế nào).
Cuốn sách MARL chuẩn (Albrecht et al., MIT Press 2024) coi non-stationarity
là MỘT khối — nên cách tách này thật sự mới.

NHƯNG PAPER CHƯA BAO GIỜ ĐO TRỰC TIẾP ĐIỀU ĐÓ. Exp1 (behavioural drift) và
Exp2 (structural shift) chạy riêng rẽ, không bao giờ đối chiếu với nhau.

Hình cần có, một biểu đồ hai đường, cùng trục tung "độ thay đổi của belief":
    - dưới BEHAVIOURAL DRIFT (cấu trúc giữ nguyên) -> belief phải PHẲNG
    - dưới STRUCTURAL SHIFT  (cấu trúc đổi thật)   -> belief phải NHẢY VỌT
    - baseline correlation-based (attention)       -> nhảy vọt ở CẢ HAI
Nếu tạo được hình này, TOÀN BỘ luận điểm nằm gọn trong một hình.

CƠ SỞ LÝ THUYẾT (mượn Pieroth ICML'24, Theorem 5.11):
    họ chứng minh TIM/SIM LIÊN TỤC theo tham số policy theta. Nghĩa là
    behavioural drift (theta đổi mượt) chỉ gây thay đổi MƯỢT trong influence.
    Vậy nếu ma trận influence NHẢY VỌT thì đó BẮT BUỘC phải là structural
    shift. Đây chính là biện minh toán học cho việc tách hai tầng —
    mượn từ một định lý đã đăng ICML.

=============================================================================
[D2] STRUCTURE SENSITIVITY — thí nghiệm PHẢI CHẠY TRƯỚC MỌI THỨ KHÁC
=============================================================================
Trong bảng kết quả của paper:
    Pure Mean Field (bỏ qua MỌI cấu trúc)      : -0.211
    Full Explicit Local (mô hình hoá MỌI THỨ)  : -0.196
    chênh lệch = 0.015

Nếu biết HẾT cấu trúc chỉ đáng giá 1.5% reward, thì TRẦN LỢI ÍCH TỐI ĐA của
BẤT KỲ phương pháp structural nào cũng chỉ trong khoảng đó. Đang thi đấu
trong một sân mà phần thưởng cho việc thắng gần bằng không.

Hàm structure_sensitivity_test() đo trần này trực tiếp bằng oracle-core
baseline. Nếu trần quá thấp -> phải sửa MÔI TRƯỜNG trước, không phải thuật toán.

=============================================================================
[D3] PROXY CALIBRATION — bảo vệ chữ "Causal" trong tên bài
=============================================================================
Tên bài có chữ "Causal" mà KHÔNG có một thí nghiệm nào kiểm chứng.
Exp3 đã được thiết kế sẵn trong paper nhưng không có số liệu.

CẢNH BÁO ĐỘ KHỚP ESTIMAND (lỗi đã phát hiện trong v1):
    proxy v1 tính  mean_a |f(a) - f(a_obs)|   (LUÔN >= 0)
    oracle env tính mean_a (R(a) - R_base)    (CÓ DẤU)
    Hai đại lượng KHÔNG THỂ khớp nhau -> Exp3 không thể pass dù thuật toán
    đúng. Hàm bên dưới BẮT BUỘC dùng effect_mode="signed_oracle_matched".
=============================================================================
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =========================================================================
# [D1] Selective responsiveness
# =========================================================================

def influence_matrix_from_beliefs(
    belief_modules: Dict[int, object],
    n_agents: int,
    signed: bool = True,
) -> np.ndarray:
    """
    Gom belief của mọi ego thành một ma trận ảnh hưởng.

    Returns:
        np.ndarray [n_agents, n_agents]
        W[i, j] = ảnh hưởng của j lên i (0 trên đường chéo)
    """
    W = np.zeros((int(n_agents), int(n_agents)), dtype=np.float64)

    for i, mod in belief_modules.items():
        for j in getattr(mod, "neighbor_ids", []):
            if int(i) == int(j):
                continue

            v = float(mod.mu_bar.get(int(j), 0.0))
            W[int(i), int(j)] = v if signed else abs(v)

    return W


def matrix_change(
    W_prev: np.ndarray,
    W_curr: np.ndarray,
    normalise: bool = True,
) -> float:
    """
    Độ thay đổi giữa hai ma trận ảnh hưởng, dùng chuẩn Frobenius.

        d = ||W_curr - W_prev||_F / (||W_prev||_F + eps)

    Vì sao đại lượng này TỐT HƠN "temporal variance" của v1:
      - v1 báo cáo ~1e-8 và diễn giải là "ổn định". Nhưng 1e-8 về bản chất
        là BẰNG KHÔNG: belief không nhúc nhích. Đó là ĐÓNG BĂNG chứ không
        phải ổn định — hai thứ khác nhau.
      - Chuẩn hoá theo ||W_prev|| cho đại lượng KHÔNG THỨ NGUYÊN, so sánh
        được giữa các phương pháp và không "nhỏ giả tạo" chỉ vì mu bé.
      - Đo trên TOÀN MA TRẬN nên bắt được cả thay đổi cấu trúc (ai ảnh hưởng
        ai) chứ không chỉ độ lớn của từng cạnh riêng lẻ.
    """
    diff = float(np.linalg.norm(W_curr - W_prev, ord="fro"))

    if not normalise:
        return diff

    base = float(np.linalg.norm(W_prev, ord="fro"))

    return diff / (base + 1e-8)


class SelectiveResponsivenessTracker:
    """
    Theo dõi độ đáp ứng của belief theo từng loại phi-dừng.

    Cách dùng:
        tracker = SelectiveResponsivenessTracker()

        # mỗi kỳ đánh giá, ở CẢ HAI chế độ môi trường:
        W = influence_matrix_from_beliefs(belief_modules, n_agents)
        tracker.record(condition="behavioural_drift", episode=ep, W=W)
        # ... chạy riêng ...
        tracker.record(condition="structural_shift", episode=ep, W=W)

        # cuối cùng:
        result = tracker.compute_selectivity()
        curves = tracker.get_curves()   # -> vẽ hình headline

    KỲ VỌNG NẾU PHƯƠNG PHÁP ĐÚNG:
        mean_change[behavioural_drift] THẤP  (cấu trúc không đổi -> đừng động)
        mean_change[structural_shift]  CAO   (cấu trúc đổi -> phải phản ứng)
        selectivity_ratio = structural / behavioural  >> 1

    Với baseline correlation-based (attention), ratio sẽ ~ 1 vì nó không
    phân biệt được hai loại. ĐÓ CHÍNH LÀ ĐIỂM CẦN CHỨNG MINH.
    """

    def __init__(self, shift_episode: Optional[int] = None):
        self.shift_episode = shift_episode
        self._history: Dict[str, List[Tuple[int, np.ndarray]]] = {}

    def record(self, condition: str, episode: int, W: np.ndarray):
        self._history.setdefault(str(condition), []).append(
            (int(episode), np.asarray(W, dtype=np.float64).copy())
        )

    def get_curves(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Returns:
            {condition: {"episodes": [...], "change": [...]}}

        `change[t]` = độ thay đổi của W từ kỳ đánh giá t-1 sang t.
        Đây là dữ liệu để vẽ HÌNH HEADLINE.
        """
        out = {}

        for cond, entries in self._history.items():
            entries = sorted(entries, key=lambda x: x[0])

            eps_list, changes = [], []

            for t in range(1, len(entries)):
                ep_prev, W_prev = entries[t - 1]
                ep_curr, W_curr = entries[t]

                eps_list.append(int(ep_curr))
                changes.append(float(matrix_change(W_prev, W_curr)))

            out[cond] = {"episodes": eps_list, "change": changes}

        return out

    def compute_selectivity(
        self,
        behavioural_key: str = "behavioural_drift",
        structural_key: str = "structural_shift",
        post_shift_only: bool = True,
    ) -> Dict[str, float]:
        """
        Con số duy nhất tóm tắt toàn bộ luận điểm của paper.

            selectivity_ratio = mean_change(structural) / mean_change(behavioural)

        >> 1  : phương pháp PHÂN BIỆT ĐƯỢC hai tầng (điều paper tuyên bố)
        ~ 1   : không phân biệt được — luận điểm trung tâm không đứng vững
        < 1   : phản ứng NGƯỢC (tệ hơn cả không làm gì)
        """
        curves = self.get_curves()

        def _mean(key: str) -> float:
            if key not in curves or len(curves[key]["change"]) == 0:
                return float("nan")

            eps = np.asarray(curves[key]["episodes"])
            ch = np.asarray(curves[key]["change"])

            if (
                post_shift_only
                and self.shift_episode is not None
                and key == structural_key
            ):
                mask = eps >= int(self.shift_episode)
                if np.any(mask):
                    ch = ch[mask]

            return float(np.mean(ch))

        b = _mean(behavioural_key)
        s = _mean(structural_key)

        ratio = (
            float(s / b)
            if (np.isfinite(b) and abs(b) > 1e-12 and np.isfinite(s))
            else float("nan")
        )

        return {
            "mean_change_behavioural": b,
            "mean_change_structural": s,
            "selectivity_ratio": ratio,
            "interpretation": (
                "GOOD: phân biệt được hai tầng" if (np.isfinite(ratio) and ratio > 1.5)
                else "WEAK: chưa phân biệt rõ" if (np.isfinite(ratio) and ratio > 1.0)
                else "FAIL: không phân biệt được / phản ứng ngược"
            ),
        }


# =========================================================================
# [D2] Structure sensitivity
# =========================================================================

def structure_sensitivity_test(
    run_fn: Callable[[str], float],
    conditions: Sequence[str] = ("pure_mean_field", "oracle_core", "full_explicit"),
    n_seeds: int = 3,
) -> Dict[str, object]:
    """
    THÍ NGHIỆM PHẢI CHẠY TRƯỚC MỌI THỨ KHÁC.

    Đo TRẦN LỢI ÍCH của việc biết cấu trúc trong môi trường này.

    Args:
        run_fn:
            hàm nhận tên condition, trả về mean reward.
            "oracle_core" phải là biến thể ĐƯỢC CHO SẴN core đúng (miễn phí,
            không cần học) — đây là cận trên lý thuyết của mọi phương pháp
            structural.
        n_seeds:
            số seed mỗi condition.

    Returns:
        dict gồm reward từng condition và structure_value.

    CÁCH ĐỌC KẾT QUẢ:
        structure_value = reward(oracle_core) - reward(pure_mean_field)

        Nếu structure_value nhỏ (ví dụ < 5% thang reward):
            -> KHÔNG THUẬT TOÁN NÀO CỨU ĐƯỢC. Phải sửa môi trường:
               tăng độ chênh lệch ảnh hưởng giữa bottleneck và agent thường,
               thắt chặt lane, tăng hình phạt tắc nghẽn, giảm số đường vòng.
            -> Đừng tốn công tối ưu thuật toán trước khi sửa việc này.

        Nếu structure_value lớn:
            -> Môi trường ổn, khoảng cách còn lại là do thuật toán.
    """
    # [FIX-5] run_fn nhận (condition, seed) nếu chấp nhận 2 tham số — bắt buộc
    # để hai condition dùng CHUNG danh sách seed (paired comparison). Bản cũ
    # chỉ nhận condition và caller tự bốc seed ngẫu nhiên => hai nhánh chạy
    # trên seed khác nhau, chênh lệch đo được lẫn cả phương sai giữa seed.
    import inspect
    try:
        takes_seed = len(inspect.signature(run_fn).parameters) >= 2
    except (TypeError, ValueError):
        takes_seed = False

    seeds = list(range(int(n_seeds)))
    results = {c: [] for c in conditions}
    for cond in conditions:
        for s in seeds:
            results[cond].append(
                float(run_fn(cond, s) if takes_seed else run_fn(cond))
            )

    summary = {
        c: {
            "mean": float(np.mean(v)) if v else float("nan"),
            "std": float(np.std(v)) if v else float("nan"),
            "n": len(v),
            "values": [float(x) for x in v],
        }
        for c, v in results.items()
    }

    # [FIX-5] Bản cũ hard-code tên "oracle_core"; sau khi run_step_0.py đổi tên
    # condition thành "explicit_local_learned" thì nhánh này KHÔNG BAO GIỜ chạy
    # => structure_value = NaN, còn verdict vẫn in ra "CẢNH BÁO: sửa MÔI TRƯỜNG"
    # một cách tự tin. Giờ lấy baseline = condition đầu, treatment = condition
    # thứ hai, không phụ thuộc tên.
    conds = list(conditions)
    base_name = "pure_mean_field" if "pure_mean_field" in summary else conds[0]
    treat_name = next((c for c in conds if c != base_name), None)

    val = float("nan")
    ci_lo = ci_hi = float("nan")
    significant = False
    if treat_name is not None:
        a = np.asarray(results[treat_name], dtype=float)
        b = np.asarray(results[base_name], dtype=float)
        if a.size and b.size:
            val = float(a.mean() - b.mean())
            # [FIX-5] Bootstrap CI: tài liệu debug yêu cầu rõ "in inconclusive
            # khi CI chứa 0" thay vì phán quyết dứt khoát trên một điểm ước
            # lượng nằm trong nhiễu.
            rng = np.random.RandomState(12345)
            diffs = [
                rng.choice(a, a.size, replace=True).mean()
                - rng.choice(b, b.size, replace=True).mean()
                for _ in range(5000)
            ]
            ci_lo, ci_hi = (float(np.percentile(diffs, 2.5)),
                            float(np.percentile(diffs, 97.5)))
            significant = bool(ci_lo > 0.0 or ci_hi < 0.0)

    if not np.isfinite(val):
        verdict = "KHÔNG ĐO ĐƯỢC — thiếu condition hợp lệ."
    elif not significant:
        verdict = (
            f"INCONCLUSIVE: CI95 = [{ci_lo:.3f}, {ci_hi:.3f}] CHỨA 0 — chưa đủ "
            f"bằng chứng kết luận môi trường có/không nhạy cấu trúc. Tăng seed "
            f"trước khi sửa môi trường."
        )
    elif val > 0:
        verdict = (
            f"Môi trường NHẠY cấu trúc (CI95 = [{ci_lo:.3f}, {ci_hi:.3f}]) — "
            f"tiếp tục cải tiến thuật toán."
        )
    else:
        verdict = (
            f"CẢNH BÁO: treatment TỆ HƠN baseline có ý nghĩa "
            f"(CI95 = [{ci_lo:.3f}, {ci_hi:.3f}])."
        )

    return {
        "per_condition": summary,
        "baseline_condition": base_name,
        "treatment_condition": treat_name,
        "structure_value": val,
        "ci95": [ci_lo, ci_hi],
        "significant": significant,
        "verdict": verdict,
        # Ghi rõ để không ai đọc nhầm lần nữa: đây KHÔNG phải oracle-core
        # zero-cost của Experiment 0 trừ khi run_fn thực sự cung cấp nó.
        "note": (
            "structure_value chỉ đúng nghĩa Experiment 0 khi treatment là "
            "oracle-core ZERO-COST (được cho sẵn core đúng, không huấn luyện). "
            "Nếu treatment là một baseline PHẢI HỌC (vd. FullExplicitLocal) "
            "thì đây là so sánh baseline-vs-baseline, không phải trần lợi ích."
        ),
    }


# =========================================================================
# [D3] Proxy calibration
# =========================================================================

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Tương quan hạng Spearman, thuần numpy (không cần scipy)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.size < 2:
        return float("nan")

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)

        # xử lý ties bằng cách lấy trung bình hạng
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts), dtype=np.float64)
        np.add.at(sums, inv, ranks)

        return (sums / counts)[inv]

    rx, ry = _rank(x), _rank(y)

    sx, sy = np.std(rx), np.std(ry)

    if sx < 1e-12 or sy < 1e-12:
        return float("nan")

    return float(np.mean((rx - np.mean(rx)) * (ry - np.mean(ry))) / (sx * sy))


def proxy_calibration_report(
    proxy_scores: np.ndarray,
    oracle_effects: np.ndarray,
    proxy_scores_baseline: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Exp3 — so proxy với oracle can thiệp thật.

    Args:
        proxy_scores:
            [N] điểm proxy. PHẢI tính bằng effect_mode="signed_oracle_matched",
            nếu không sẽ so nhầm đại lượng (xem cảnh báo đầu file).
        oracle_effects:
            [N] hiệu ứng can thiệp thật, CÓ DẤU, từ
            OracleInterventionSampler.signed_effect().
        proxy_scores_baseline:
            [N] tuỳ chọn — điểm từ effect_mode="range" (kiểu Pieroth,
            KHÔNG DẤU). Nếu bản có dấu thắng bản này, đó là bằng chứng
            TRỰC TIẾP cho novelty.

    Returns:
        dict với bias, rank correlation, sign agreement...

    NGƯỠNG DIỄN GIẢI:
        spearman > 0.3 và ổn định qua seed -> đủ để dùng cho xếp hạng cấu trúc,
                                              được quyền giữ chữ "Causal".
        spearman ~ 0                        -> proxy vô dụng, PHẢI đổi tên bài.
        sign_agreement > 0.7                -> tín hiệu có dấu đáng tin,
                                              slot Thiện/Ác có căn cứ.
    """
    p = np.asarray(proxy_scores, dtype=np.float64).reshape(-1)
    o = np.asarray(oracle_effects, dtype=np.float64).reshape(-1)

    n = int(min(p.size, o.size))

    if n < 2:
        return {"n": n, "error": "không đủ mẫu"}

    p, o = p[:n], o[:n]

    bias = float(np.mean(p - o))
    mae = float(np.mean(np.abs(p - o)))

    pearson = float("nan")
    if np.std(p) > 1e-12 and np.std(o) > 1e-12:
        pearson = float(np.corrcoef(p, o)[0, 1])

    spearman = _spearman(p, o)

    # Chỉ tính đồng thuận dấu trên các mẫu mà oracle thật sự khác 0.
    mask = np.abs(o) > 1e-8
    sign_agreement = (
        float(np.mean(np.sign(p[mask]) == np.sign(o[mask])))
        if np.any(mask)
        else float("nan")
    )

    out = {
        "n": n,
        "bias": bias,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "sign_agreement": sign_agreement,
        "proxy_std": float(np.std(p)),
        "oracle_std": float(np.std(o)),
    }

    if proxy_scores_baseline is not None:
        b = np.asarray(proxy_scores_baseline, dtype=np.float64).reshape(-1)[:n]

        # Baseline không dấu -> so với |oracle| mới công bằng.
        out["baseline_spearman_vs_abs_oracle"] = _spearman(b, np.abs(o))
        out["signed_spearman_vs_signed_oracle"] = spearman
        out["signed_beats_unsigned"] = bool(
            np.isfinite(spearman)
            and np.isfinite(out["baseline_spearman_vs_abs_oracle"])
            and abs(spearman) > abs(out["baseline_spearman_vs_abs_oracle"])
        )

    out["verdict"] = (
        "PASS: proxy đủ tốt cho xếp hạng cấu trúc"
        if np.isfinite(spearman) and spearman > 0.3
        else "WEAK: tương quan yếu, cần thêm mẫu can thiệp"
        if np.isfinite(spearman) and spearman > 0.1
        else "FAIL: proxy không tương quan với can thiệp thật"
    )

    return out


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Khoảng tin cậy bootstrap — paper hiện KHÔNG có kiểm định thống kê nào.

    Vì sao BẮT BUỘC: trong bảng kết quả,
        Final CIG-AMF  -0.199 +- 0.018
        Pure MF        -0.211 +- 0.008
    chênh 0.012 trong khi std là 0.018, với 5 seed. Đây KHÔNG phải khác biệt
    có ý nghĩa. Mọi phát biểu kiểu "moved into the strongest reward group"
    hiện chưa có căn cứ.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)

    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    rng = np.random.RandomState(int(seed))
    means = np.array([
        np.mean(rng.choice(v, size=v.size, replace=True))
        for _ in range(int(n_boot))
    ])

    return {
        "mean": float(np.mean(v)),
        "lo": float(np.percentile(means, 100.0 * alpha / 2.0)),
        "hi": float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0))),
        "n": int(v.size),
    }


def compare_two_methods(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    So hai phương pháp bằng bootstrap trên HIỆU SỐ.

    Returns:
        prob_a_better: xác suất A tốt hơn B theo phân phối bootstrap.
        significant  : True nếu khoảng tin cậy của hiệu không chứa 0.

    Báo cáo con số này thay vì chỉ nói "A cao hơn B".
    """
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)

    if a.size == 0 or b.size == 0:
        return {"error": "mảng rỗng"}

    rng = np.random.RandomState(int(seed))

    diffs = np.array([
        np.mean(rng.choice(a, a.size, replace=True))
        - np.mean(rng.choice(b, b.size, replace=True))
        for _ in range(int(n_boot))
    ])

    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))

    return {
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "ci_lo": lo,
        "ci_hi": hi,
        "prob_a_better": float(np.mean(diffs > 0.0)),
        "significant": bool(lo > 0.0 or hi < 0.0),
    }
