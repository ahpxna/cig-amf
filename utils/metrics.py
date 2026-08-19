import numpy as np
from scipy.stats import spearmanr


def core_f1(pred_core, gt_core):
    """
    F1 cho core set identification.

    pred_core:
        iterable neighbour ids predicted as core

    gt_core:
        iterable neighbour ids in diagnostic/ground-truth core
    """
    pred = set(pred_core)
    gt = set(gt_core)

    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def safe_spearman(x, y):
    """Spearman rank correlation does not issue warning when vector is constant.

A constant vector is valid in the tiny oracle or before the proxy has learned.
Record constant_case=1 instead of allowing SciPy to emit a warning."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(x) <= 1 or len(y) <= 1:
        return 0.0, 1.0, 1

    x_const = bool(np.allclose(x, x[0]))
    y_const = bool(np.allclose(y, y[0]))

    if x_const or y_const:
        return 0.0, 1.0, 1

    rho, p_value = spearmanr(x, y)

    if np.isnan(rho):
        rho = 0.0
    if np.isnan(p_value):
        p_value = 1.0

    return float(rho), float(p_value), 0


def oracle_calibration(learned_scores, oracle_scores, neighbor_ids):
    """Calibration summary between learned proxy score and oracle intervention score.

learned_scores:
    dict {neighbor_id: learned score}

oracle_scores:
    dict {neighbor_id: oracle intervention score}

neighbor_ids:
    list ids to compare

Return:
    bias
    variance
    mae
    rmse
    rank_correlation
    p_value
    constant_case

Note:
    Use abs(score) because core selection is based on magnitude of influence.
    If signed agreement is desired, add a separate metric."""
    neighbor_ids = list(neighbor_ids)

    learned = np.array(
        [abs(float(learned_scores.get(j, 0.0))) for j in neighbor_ids],
        dtype=np.float32,
    )
    oracle = np.array(
        [abs(float(oracle_scores.get(j, 0.0))) for j in neighbor_ids],
        dtype=np.float32,
    )

    if len(neighbor_ids) == 0:
        return {
            "bias": 0.0,
            "variance": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "rank_correlation": 0.0,
            "p_value": 1.0,
            "constant_case": 1,
        }

    diff = learned - oracle
    rho, p_value, constant_case = safe_spearman(learned, oracle)

    return {
        "bias": float(np.mean(diff)),
        "variance": float(np.var(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "rank_correlation": float(rho),
        "p_value": float(p_value),
        "constant_case": int(constant_case),
    }


def oracle_core_f1_from_scores(learned_scores, oracle_scores, neighbor_ids, top_k):
    """F1 between top-k learned neighbours and top-k oracle neighbours.

Use magnitude because CIG-AMF selects core based on strength of influence,
not just positive influence."""
    neighbor_ids = list(neighbor_ids)
    top_k = int(max(0, top_k))

    if top_k == 0 or len(neighbor_ids) == 0:
        return 0.0

    learned_ranked = sorted(
        neighbor_ids,
        key=lambda j: abs(float(learned_scores.get(j, 0.0))),
        reverse=True,
    )
    oracle_ranked = sorted(
        neighbor_ids,
        key=lambda j: abs(float(oracle_scores.get(j, 0.0))),
        reverse=True,
    )

    pred = set(learned_ranked[:top_k])
    gt = set(oracle_ranked[:top_k])

    return core_f1(pred, gt)


def recovery_latency(
    history,
    shift_eval_index,
    metric_key="mean_reward",
    threshold=0.90,
    lookback=2,
    horizon=6,
    higher_is_better=True,
):
    """Recovery latency per evaluation index.

This version modifies the old logic target = threshold * baseline, because environment reward can be negative.
With negative reward, baseline=-0.4 then 0.9*baseline=-0.36, which looks very misleading.

New logic:
    baseline = mean metric before shift
    post_shift_extreme = worst point in window after shift

If higher_is_better=True:
    target = post_shift_min + threshold * (baseline - post_shift_min)
    recovered when metric >= target

If higher_is_better=False:
    target = post_shift_max - threshold * (post_shift_max - baseline)
    recovered when metric <= target

Return:
    latency, baseline, target

latency:
    number of evaluation points from shift_eval_index to first point reaching target.
    If not recovered within horizon, latency = -1."""
    xs = history.get(metric_key, [])
    xs = [float(x) for x in xs]

    if len(xs) == 0:
        return -1, None, None

    shift_eval_index = int(shift_eval_index)
    lookback = int(lookback)
    horizon = int(horizon)
    threshold = float(threshold)

    if shift_eval_index <= 0 or shift_eval_index >= len(xs):
        return -1, None, None

    left = max(0, shift_eval_index - lookback)

    if shift_eval_index <= left:
        return -1, None, None

    baseline = float(np.mean(xs[left:shift_eval_index]))

    right = min(len(xs), shift_eval_index + horizon + 1)
    post = xs[shift_eval_index:right]

    if len(post) == 0:
        return -1, baseline, None

    if higher_is_better:
        post_shift_extreme = float(np.min(post))
        target = float(post_shift_extreme + threshold * (baseline - post_shift_extreme))

        for k in range(shift_eval_index, right):
            if float(xs[k]) >= target:
                return int(k - shift_eval_index), baseline, target

    else:
        post_shift_extreme = float(np.max(post))
        target = float(post_shift_extreme - threshold * (post_shift_extreme - baseline))

        for k in range(shift_eval_index, right):
            if float(xs[k]) <= target:
                return int(k - shift_eval_index), baseline, target

    return -1, baseline, target


def smooth_curve(xs, k=3):
    xs = list(xs)
    k = int(k)

    if len(xs) < k or k <= 1:
        return xs

    out = []

    for i in range(len(xs)):
        left = max(0, i - k + 1)
        out.append(float(np.mean(xs[left:i + 1])))

    return out


def summarize_final_window(history, metric_key, window=3):
    """Take mean/std of some final evaluation points.
Used when reporting final performance table."""
    xs = history.get(metric_key, [])

    if xs is None or len(xs) == 0:
        return {
            f"{metric_key}_final_mean": 0.0,
            f"{metric_key}_final_std": 0.0,
            f"{metric_key}_final_n": 0,
        }

    vals = np.asarray(xs[-int(window):], dtype=np.float32)

    return {
        f"{metric_key}_final_mean": float(np.mean(vals)),
        f"{metric_key}_final_std": float(np.std(vals)),
        f"{metric_key}_final_n": int(len(vals)),
    }


def aggregate_seed_runs(rows, group_keys=("task", "model", "n_agents"), metric_keys=None):
    """Aggregate multiple seeds from list[dict] results.

rows:
    list of dict, each row is typically the final summary of one seed.

group_keys:
    keys to group by, e.g. task/model/n_agents.

metric_keys:
    if None, automatically take numeric keys not in group_keys.

Return:
    list[dict] containing mean/std/n per group."""
    rows = list(rows)

    if len(rows) == 0:
        return []

    group_keys = tuple(group_keys)

    if metric_keys is None:
        metric_keys = []
        for row in rows:
            for k, v in row.items():
                if k in group_keys:
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)):
                    if k not in metric_keys:
                        metric_keys.append(k)

    groups = {}

    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(row)

    out = []

    for key, group_rows in groups.items():
        out_row = {}

        for k, v in zip(group_keys, key):
            out_row[k] = v

        out_row["n"] = int(len(group_rows))

        for metric in metric_keys:
            vals = []

            for row in group_rows:
                if metric in row:
                    try:
                        vals.append(float(row[metric]))
                    except Exception:
                        pass

            if len(vals) == 0:
                continue

            vals = np.asarray(vals, dtype=np.float32)
            out_row[f"{metric}_mean"] = float(np.mean(vals))
            out_row[f"{metric}_std"] = float(np.std(vals))

        out.append(out_row)

    return out
