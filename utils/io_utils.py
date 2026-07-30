import os
import csv
import json
import matplotlib.pyplot as plt


def ensure_dir(path):
    """
    Tạo thư mục nếu path hợp lệ.

    path có thể là:
    - folder thật
    - "" nếu file nằm ở current directory
    - None trong vài nhánh gọi phụ
    """
    if path is None:
        return

    path = str(path)
    if path.strip() == "":
        return

    os.makedirs(path, exist_ok=True)


def _union_keys(rows):
    """
    Lấy union keys theo thứ tự xuất hiện.

    Không dùng keys của row đầu tiên vì mỗi runner có thể có thêm metric riêng.
    Ví dụ Final-CIGAMF có proxy_buffer_size, PureMeanField thì không.
    """
    keys = []
    seen = set()

    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    return keys


def save_csv(rows, path):
    """
    Lưu list[dict] thành CSV.

    Nếu rows rỗng, vẫn tạo file rỗng để biết experiment đã chạy tới nhánh đó.
    """
    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")

    if rows is None or len(rows) == 0:
        with open(path, "w", newline="") as f:
            f.write("")
        return

    keys = _union_keys(rows)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_history_csv(hist, path, extra=None):
    """
    Lưu history per evaluation point.

    hist:
        dict[str, list]

    extra:
        metadata gắn vào mỗi row:
            experiment, model, seed, n_agents, episodes, max_steps, ...
    """
    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")

    extra = extra or {}

    if hist is None or len(hist) == 0:
        save_csv([], path)
        return

    list_keys = [k for k, v in hist.items() if isinstance(v, list)]

    if len(list_keys) == 0:
        row = dict(extra)
        save_csv([row], path)
        return

    n = max(len(hist[k]) for k in list_keys)
    rows = []

    for i in range(n):
        row = dict(extra)

        for k in list_keys:
            values = hist[k]
            row[k] = values[i] if i < len(values) else ""

        rows.append(row)

    save_csv(rows, path)


def plot_histories(histories, metric, title, save_path):
    """
    Vẽ metric theo evaluation episode.

    Không crash nếu một số baseline không có metric đó.
    """
    ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else ".")

    plt.figure(figsize=(8, 4.5))

    plotted = False

    for name, hist in histories.items():
        if hist is None:
            continue
        if metric not in hist:
            continue
        if "episodes" not in hist:
            continue

        xs = hist["episodes"]
        ys = hist[metric]

        if len(xs) == 0 or len(ys) == 0:
            continue

        n = min(len(xs), len(ys))
        plt.plot(xs[:n], ys[:n], marker="o", label=name)
        plotted = True

    plt.title(title)
    plt.xlabel("evaluation episode")
    plt.ylabel(metric)

    if plotted:
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()