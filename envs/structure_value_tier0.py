"""
Tầng 0 — Hối tiếc Oracle. Không huấn luyện.
Đo trần lý thuyết của giá trị cấu trúc: biết chính xác core đang làm gì
thì tốt hơn bao nhiêu so với chỉ biết thống kê trung bình.

Sửa (xem báo cáo triển khai của người dùng, Task #11/#12/#14):
- oracle_structure_regret() giờ so sánh CHUỖI hành động k-bước (k_seq, mặc
  định k=2) của ego, không phải chỉ một hành động đơn -- để oracle có thể
  phát hiện được các chiến lược thật sự (vd. "đợi 1 bước rồi mới tiến" vs
  "tiến ngay") mà một phép hối tiếc một-bước không thể thấy.
- _rollout_custom() (hỏng -- NameError trên `seq`/`ego` không tồn tại, sai
  thứ tự tham số, trả về scalar nhưng bị subscript như dict) được thay bằng
  _rollout_seq(), dùng ĐÚNG API thật của OmniArena/TinyOracleDIG:
  env.clone_state()/restore_state(), env.scripted_policy(i), env.step(acts)
  -- không có method env.forced_step()/env.rollout_seq() nào tồn tại trong
  omni_arena.py, nên ta tự dựng acts[] mỗi bước rồi gọi step() trực tiếp,
  đúng khuôn mẫu oracle_w_star_sustained() trong env_audit.py. Tái sử dụng
  hạ tầng CRN (set_noise_buffer/_make_crn_buffer/clear_noise_buffer, P0d)
  để so sánh các nhánh (36 chuỗi x q_full/q_blind) trên cùng một luồng
  nhiễu, giảm phương sai không liên quan tới can thiệp.
- q_range (max-min) làm mẫu số chuẩn hoá bị nhiễu bởi hành động/chuỗi TỆ
  NHẤT (không bao giờ được chọn) -- tăng một penalty bất kỳ có thể kéo
  min_a Q xuống, làm q_range phình to giả tạo, khiến normalised_regret bị
  pha loãng dù cấu trúc không hề trở nên "đáng biết hơn". Giữ q_range/
  normalised_regret cho khả năng so sánh ngược, nhưng bổ sung std(Q) +
  norm_by_std làm chỉ số chính, cùng các thống kê "| đã đổi hành động".
"""

import itertools

import numpy as np

from omni_arena import OmniArena


def _rollout_seq(
    env, ego_id, seq, forced_others, horizon, discount, n_trials, rng,
    crn_seed=None,
):
    """
    Rollout ego-scalar: ego bị ép theo `seq` (list/tuple độ dài k) trong k
    bước ĐẦU của horizon, sau đó ego quay lại scripted_policy() cho phần
    còn lại. `forced_others`: dict {agent_id: action} bị ép SUỐT horizon
    (dùng để mô phỏng "core bị thay bằng hành động hàng xóm trung bình" cho
    q_blind; {} rỗng cho q_full). Mọi agent khác luôn dùng
    env.scripted_policy(i) -- không có method forced_step()/rollout_seq()
    nào tồn tại trên OmniArena, nên acts[] được dựng thủ công mỗi bước rồi
    gọi env.step(acts) trực tiếp, giống hệt khuôn mẫu
    oracle_w_star_sustained() trong env_audit.py.

    Trả về: float — discounted return CỦA RIÊNG ego, trung bình qua
    n_trials, dùng chung một buffer CRN (P0d) giữa các trial để giảm
    phương sai không liên quan tới can thiệp khi so sánh nhiều chuỗi/nhiều
    nhánh (q_full vs q_blind) với nhau.
    """
    saved = env.clone_state()
    k = len(seq)
    seed_base = int(crn_seed) if crn_seed is not None else int(env.seed)

    accs = []
    for trial in range(int(n_trials)):
        crn_rng = np.random.RandomState((seed_base * 9973 + trial) % (2**32))
        buffer = env._make_crn_buffer(horizon, crn_rng)

        env.restore_state(saved)
        env.set_noise_buffer(buffer)

        total, g = 0.0, 1.0
        done = False
        for t in range(int(horizon)):
            acts = [int(env.scripted_policy(i)) for i in range(env.n_agents)]
            if t < k:
                acts[ego_id] = int(seq[t])
            for j, a in forced_others.items():
                acts[j] = int(a)

            _, rew, done, _ = env.step(acts)
            total += g * float(rew[ego_id])
            g *= discount
            if done:
                break

        env.clear_noise_buffer()
        accs.append(total)

    env.restore_state(saved)
    return float(np.mean(accs))


def _marginal_action_dist(env, n_samples=200, rng=None):
    rng = rng or np.random.RandomState(0)
    A = env.get_action_dim()
    counts = np.zeros(A, dtype=np.float64)

    saved = env.clone_state()
    for _ in range(n_samples):
        try:
            acts = [int(env.scripted_policy(i)) for i in range(env.n_agents)]
            for act in acts:
                counts[act] += 1.0
        except AttributeError:
            acts = [int(rng.randint(A)) for _ in range(env.n_agents)]
            for act in acts:
                counts[act] += 1.0
        env.step(acts)
    env.restore_state(saved)

    return counts / max(1.0, counts.sum())


def oracle_structure_regret(
    env, ego_id, k_core=3, horizon=8, discount=0.95,
    n_trials=3, n_mf_samples=8, rng=None, k_seq=2,
):
    rng = rng or np.random.RandomState(0)
    A = env.get_action_dim()
    N = env.n_agents
    saved = env.clone_state()

    # ---- 1. Core thật: top-k theo |W*| ----
    # LƯU Ý: intervention_action PHẢI là một hành động thật (env.STAY), KHÔNG
    # phải None -- compute_oracle_influence_from_current_state() coi
    # candidate_actions=None mặc định là range(N_ACTIONS), rồi CHỈ append
    # intervention_action nếu nó "not in candidate_actions"; None không nằm
    # trong [0..5] nên sẽ bị append THẲNG vào candidate_actions, biến thành
    # một "hành động" giả (forced_action=None) được rollout thật sự và trộn
    # vào per_action/signed/range/best/worst -- không crash, nhưng làm bẩn
    # hồ sơ ảnh hưởng một cách âm thầm. Dùng env.STAY (đã có trong
    # range(N_ACTIONS)) để candidate_actions không bị thêm phần tử rác.
    infl = {}
    for j in range(N):
        if j == ego_id:
            continue
        try:
            profile = env.compute_oracle_influence_from_current_state(
                ego_id=ego_id, agent_j=j, intervention_action=env.STAY,
                horizon=horizon, n_trials=n_trials,
            )
            infl[j] = abs(float(profile["signed"]))
        except Exception:
            infl[j] = 0.0
        env.restore_state(saved)

    core_ids = sorted(infl, key=infl.get, reverse=True)[:int(k_core)]

    # ---- 2. Liệt kê toàn bộ chuỗi k_seq-bước trên A hành động ----
    # k_seq=2, A=6 -> 36 chuỗi (khớp Task #11: k=2, |A|=6).
    sequences = list(itertools.product(range(A), repeat=int(k_seq)))
    n_seq = len(sequences)

    # Một crn_seed CHUNG cho toàn bộ lệnh gọi này (mọi chuỗi, cả q_full lẫn
    # q_blind) -- để 36 chuỗi được so sánh trên đúng cùng một luồng nhiễu
    # resource_respawn, không phải 36 luồng ngẫu nhiên độc lập.
    crn_seed = int(rng.randint(1, 2 ** 31 - 1))

    # ---- 3. Q_full(seq): biết rõ mọi thứ (không thay thế core) ----
    q_full = np.zeros(n_seq, dtype=np.float64)
    for si, seq in enumerate(sequences):
        env.restore_state(saved)
        q_full[si] = _rollout_seq(
            env, ego_id, seq, forced_others={}, horizon=horizon,
            discount=discount, n_trials=n_trials, rng=rng, crn_seed=crn_seed,
        )

    # ---- 4. Q_blind(seq): core bị thay bằng "hàng xóm trung bình" ----
    marg = _marginal_action_dist(env, rng=rng)
    q_blind = np.zeros(n_seq, dtype=np.float64)

    for si, seq in enumerate(sequences):
        vals = []
        for _ in range(int(n_mf_samples)):
            forced_others = {j: int(rng.choice(A, p=marg)) for j in core_ids}
            env.restore_state(saved)
            val = _rollout_seq(
                env, ego_id, seq, forced_others=forced_others, horizon=horizon,
                discount=discount, n_trials=1, rng=rng, crn_seed=crn_seed,
            )
            vals.append(val)
        q_blind[si] = float(np.mean(vals))

    env.restore_state(saved)

    a_full = int(np.argmax(q_full))
    a_blind = int(np.argmax(q_blind))
    # Hối tiếc = giá trị THẬT (q_full) của lựa chọn tối ưu thật, trừ giá trị
    # THẬT (vẫn q_full, không phải q_blind) của lựa chọn mà agent "mù cấu
    # trúc" sẽ chọn -- đây là chi phí cơ hội thật của việc không biết core,
    # đánh giá dưới đúng động lực thật (q_full), không phải dưới ước lượng
    # mean-field của chính nó.
    regret = float(q_full[a_full] - q_full[a_blind])

    q_std = float(np.std(q_full))
    q_median = float(np.median(q_full))

    return {
        "regret": regret,
        "q_best_full": float(q_full[a_full]),
        "q_chosen_blind": float(q_full[a_blind]),
        "q_range": float(q_full.max() - q_full.min()),
        "q_std": q_std,
        "q_max_minus_median": float(q_full.max() - q_median),
        "core_ids": core_ids,
        "best_seq_full": sequences[a_full],
        "best_seq_blind": sequences[a_blind],
        "action_changed": bool(a_full != a_blind),
    }


def run_tier0(
    env, n_states=30, steps_between=8, k_core=3, horizon=8, egos=None,
    seed=0, k_seq=2,
):
    rng = np.random.RandomState(seed)
    egos = list(range(env.n_agents)) if egos is None else list(egos)

    rows = []
    env.reset()

    for s in range(int(n_states)):
        # Tránh lỗi vượt quá episode boundary (P2<->P4 trap)
        if env.t + horizon + steps_between >= env.max_steps:
            env.reset()

        for _ in range(int(steps_between)):
            env.step([int(env.scripted_policy(i)) for i in range(env.n_agents)])

        for ego in egos:
            rows.append(oracle_structure_regret(
                env, ego, k_core=k_core, horizon=horizon, rng=rng, k_seq=k_seq,
            ))

    regrets = np.array([r["regret"] for r in rows])
    ranges = np.array([r["q_range"] for r in rows])
    q_std = np.array([r["q_std"] for r in rows])
    changed = np.array([r["action_changed"] for r in rows])

    live = ranges > 1e-9
    live_std = q_std > 1e-9
    norm = float(np.mean(regrets[live] / ranges[live])) if live.any() else 0.0
    norm_by_std = (
        float(np.mean(regrets[live_std] / q_std[live_std]))
        if live_std.any() else float("nan")
    )

    mask = changed
    regret_given_changed = float(regrets[mask].mean()) if mask.any() else float("nan")

    mask_std = mask & live_std
    norm_given_changed_by_std = (
        float(np.mean(regrets[mask_std] / q_std[mask_std]))
        if mask_std.any() else float("nan")
    )

    p95_regret = float(np.percentile(regrets, 95)) if len(regrets) else float("nan")

    out = {
        "mean_regret": float(regrets.mean()),
        "median_regret": float(np.median(regrets)),
        "p90_regret": float(np.percentile(regrets, 90)),
        "p95_regret": p95_regret,
        "normalised_regret": norm,               # q_range-based (bị nhiễu, xem docstring module)
        "norm_by_std": norm_by_std,               # std(Q)-based, chỉ số chuẩn hoá CHÍNH
        "frac_action_changed": float(changed.mean()),
        "regret_given_changed": regret_given_changed,
        "norm_given_changed_by_std": norm_given_changed_by_std,
        "n_samples": int(len(rows)),
    }

    def _fmt(x):
        return "NaN (không có mẫu action_changed)" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"

    print(f"""
TẦNG 0 — HỐI TIẾC ORACLE  (n={out['n_samples']}, k_seq={k_seq})
  Hối tiếc trung bình              = {out['mean_regret']:.4f}
  Hối tiếc trung vị                = {out['median_regret']:.4f}
  P90 hối tiếc                     = {out['p90_regret']:.4f}
  P95 hối tiếc                     = {out['p95_regret']:.4f}
  Hối tiếc chuẩn hoá (q_range)     = {out['normalised_regret']:.3f}   [LƯU Ý: mẫu số bị nhiễu bởi chuỗi TỆ NHẤT -- xem norm_by_std]
  Hối tiếc chuẩn hoá (std Q)       = {_fmt(out['norm_by_std'])}
  Tỷ lệ ĐỔI chuỗi hành động        = {out['frac_action_changed']:.1%}
  Hối tiếc TB | đã đổi             = {_fmt(out['regret_given_changed'])}
  Hối tiếc chuẩn hoá(std) | đã đổi = {_fmt(out['norm_given_changed_by_std'])}

  Đọc kết quả:
    Tỷ lệ đổi hành động < 10%  -> Cấu trúc gần như KHÔNG ảnh hưởng quyết định.
    Hối tiếc chuẩn hoá(std) < 0.05  -> Biết cấu trúc gần như vô ích ở đây.
    Cả hai đều cao             -> Cấu trúc đáng biết, chạy tiếp Tầng 1-2.
    (Nếu frac_action_changed == 0, các thống kê "| đã đổi" là NaN -- không
     phải lỗi, chỉ là không có mẫu nào để tính trên đó.)
""")
    return out, rows


if __name__ == "__main__":
    env = OmniArena(
        n_agents=24,
        grid_size=24,
        n_zones=4,
        enable_conditional_gates=True,
        enable_latency_ladder=True,
        enable_congestion=True,
        enable_structural_shift=True
    )

    print("Bắt đầu lấy mẫu Tầng 0 (dự kiến ~20 phút)...")
    out, rows = run_tier0(env=env, n_states=20, steps_between=10, k_core=3, horizon=8, seed=123)
