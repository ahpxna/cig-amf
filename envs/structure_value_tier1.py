# structure_value_tier1.py
import numpy as np
import torch
import torch.nn as nn


def collect_dataset(env, n_episodes=40, horizon=8, discount=0.95,
                    k_core=3, seed=0):
    """
    Thu thập (obs_i, core_actions, mean_action, R_i^(H)).

    Ba sửa lỗi so với bản gốc (xem review kết quả MSE_core > MSE_solo/mf,
    frac_from_core = -88 nghìn tỷ %):

    (a) Hành động thu thập dữ liệu dùng scripted_policy + 15% nhiễu ngẫu
        nhiên, KHÔNG phải random thuần -- dưới random thuần, gatekeeper gần
        như không bao giờ mở cổng đúng nhịp nên không có cấu trúc thật để
        phát hiện (Tier 0 dùng scripted_policy và thấy tín hiệu thật; bản
        gốc của Tier 1 đo một môi trường "không cấu trúc" khác hẳn).

    (b) Đặc trưng mean-field được TILE lại k_core lần (mf_matched, cùng số
        chiều k_core*A với core one-hot) trước khi đưa vào bộ so sánh MSE,
        để tránh core (nhiều chiều hơn) overfit "miễn phí" chỉ vì có nhiều
        tham số hơn với cùng lượng dữ liệu.

    (c) Core KHÔNG còn được xác định một lần duy nhất (giữa episode) rồi
        dùng lại cho toàn bộ episode. T3 CV=1.035 (đã đo, xem báo cáo) xác
        nhận độ ảnh hưởng phụ thuộc trạng thái rất mạnh (delta_ij(s), Phần
        2.1) -- core "đúng" ở đầu/giữa/cuối episode nhiều khả năng khác
        nhau. Episode được chia làm 3 đoạn (thirds theo T=len(traj)); core
        được tính riêng cho mỗi đoạn bằng oracle tại một mốc đại diện
        (midpoint) của đoạn đó; khi tạo mẫu huấn luyện cho bước t, dùng core
        của ĐÚNG đoạn chứa t.
    """
    rng = np.random.RandomState(seed)
    A = env.get_action_dim()
    N = env.n_agents

    X_obs, X_core, X_mf, Y = [], [], [], []

    for ep in range(n_episodes):
        env.reset()
        traj = []

        # Ba mốc đại diện (~1/6, 1/2, 5/6 của max_steps) cho 3 đoạn
        # đầu/giữa/cuối episode, mỗi mốc được kẹp <= max_steps - horizon để
        # oracle luôn còn đủ chỗ rollout horizon bước (assert P2<->P4 trap
        # guard: self.t + horizon <= self.max_steps).
        safe_cap = max(0, env.max_steps - horizon)
        seg_targets = [
            min(safe_cap, max(0, int(env.max_steps * frac)))
            for frac in (1 / 6, 1 / 2, 5 / 6)
        ]
        snapshots = [None, None, None]

        for t in range(env.max_steps):
            acts = [int(env.scripted_policy(a)) if rng.rand() > 0.15
                    else int(rng.randint(A)) for a in range(N)]
            obs, rew, done, _ = env.step(acts)
            traj.append((obs, acts, rew))
            for si in range(3):
                if snapshots[si] is None and env.t >= seg_targets[si]:
                    snapshots[si] = env.clone_state()
            if done:
                break

        # Nếu episode kết thúc sớm hơn một mốc (không nên xảy ra vì
        # max_steps là cố định và done chỉ true khi t>=max_steps, nhưng
        # phòng thủ vẫn xử lý): dùng snapshot hợp lệ gần nhất trước đó, hoặc
        # trạng thái hiện tại nếu chưa có snapshot nào.
        last_valid = None
        for si in range(3):
            if snapshots[si] is None:
                snapshots[si] = last_valid if last_valid is not None else env.clone_state()
            else:
                last_valid = snapshots[si]

        # core thật, xác định RIÊNG cho từng đoạn của episode.
        core_by_ego_segs = []
        for si in range(3):
            env.restore_state(snapshots[si])
            core_by_ego = {}
            for ego in range(N):
                # intervention_action PHẢI là hành động thật (env.STAY),
                # không phải None: compute_oracle_influence_from_current_state()
                # chỉ thêm intervention_action vào candidate_actions nếu nó
                # "not in candidate_actions" -- None không nằm trong
                # range(N_ACTIONS) nên bị append thẳng vào, tạo ra một
                # "hành động" rollout giả (forced_action=None) trộn vào
                # per_action/signed -- không crash, nhưng làm bẩn hồ sơ
                # ảnh hưởng.
                infl = {j: abs(float(env.compute_oracle_influence_from_current_state(
                            ego_id=ego, agent_j=j, intervention_action=env.STAY,
                            horizon=horizon, n_trials=1)))
                        for j in range(N) if j != ego}
                core_by_ego[ego] = sorted(infl, key=infl.get, reverse=True)[:k_core]
            core_by_ego_segs.append(core_by_ego)

        T = len(traj)
        seg_bounds = [0, T // 3, 2 * T // 3, T]

        for t in range(T):
            # xác định đoạn chứa bước t
            if t < seg_bounds[1]:
                seg_idx = 0
            elif t < seg_bounds[2]:
                seg_idx = 1
            else:
                seg_idx = 2
            core_by_ego = core_by_ego_segs[seg_idx]

            # return H bước
            R = np.zeros(N)
            for h in range(horizon):
                if t + h < T:
                    R += (discount ** h) * np.asarray(traj[t + h][2])

            obs_t, acts_t, _ = traj[t]

            for ego in range(N):
                core = core_by_ego[ego]
                core_oh = np.zeros(k_core * A, dtype=np.float32)
                for idx, j in enumerate(core):
                    core_oh[idx * A + int(acts_t[j])] = 1.0

                others = [acts_t[j] for j in range(N) if j != ego]
                mf = np.bincount(others, minlength=A).astype(np.float32)
                mf /= max(1.0, mf.sum())
                # (b) tile mean-field lên cùng số chiều với core one-hot
                # (k_core*A) để so sánh MSE công bằng về số tham số.
                mf_matched = np.tile(mf, k_core).astype(np.float32)

                X_obs.append(np.asarray(obs_t[ego], dtype=np.float32))
                X_core.append(core_oh)
                X_mf.append(mf_matched)
                Y.append(float(R[ego]))

    return (np.stack(X_obs), np.stack(X_core),
            np.stack(X_mf), np.asarray(Y, dtype=np.float32))


def _fit(X, Y, hidden=128, epochs=60, seed=0):
    """Hồi quy MLP, trả MSE trên tập kiểm tra."""
    torch.manual_seed(seed)
    n = len(Y)
    idx = np.random.RandomState(seed).permutation(n)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]

    Xt = torch.tensor(X[tr]); Yt = torch.tensor(Y[tr])
    Xe = torch.tensor(X[te]); Ye = torch.tensor(Y[te])

    net = nn.Sequential(
        nn.Linear(X.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    for _ in range(epochs):
        perm = torch.randperm(len(Yt))
        for b in range(0, len(Yt), 256):
            sel = perm[b:b + 256]
            loss = nn.functional.mse_loss(net(Xt[sel]).squeeze(-1), Yt[sel])
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        return float(nn.functional.mse_loss(net(Xe).squeeze(-1), Ye))


def run_tier1(env, **kw):
    Xo, Xc, Xm, Y = collect_dataset(env, **kw)

    mse_core = _fit(np.concatenate([Xo, Xc], axis=1), Y)
    mse_mf   = _fit(np.concatenate([Xo, Xm], axis=1), Y)
    mse_solo = _fit(Xo, Y)

    # Bao nhiêu phần thông tin mà HÀNG XÓM mang lại đến từ việc biết
    # ĐÍCH DANH core, thay vì chỉ biết thống kê đám đông?
    gain_total = mse_solo - mse_core       # tất cả thông tin hàng xóm
    gain_mf    = mse_solo - mse_mf         # phần mean-field đã bắt được
    gain_core  = mse_mf - mse_core         # PHẦN THÊM do biết core

    # (d) frac = gain_core / gain_total có thể chia cho một số RẤT nhỏ hoặc
    # ÂM khi gain_total <= 0 (core dự đoán TỆ HƠN solo) -- max(1e-12, .) ép
    # mẫu số về epsilon dương bất kể dấu thật của gain_total, sinh ra tỷ lệ
    # vô nghĩa (vd. -88 nghìn tỷ %). Không còn ép sàn: nếu |gain_total| quá
    # nhỏ để chia có ý nghĩa, báo NaN thay vì bịa số.
    denom = gain_total if abs(gain_total) > 1e-9 else float("nan")
    frac = gain_core / denom

    # R² là chỉ số phải xem TRƯỚC TIÊN: nếu cả ba mô hình đều có R² ≈ 0 thì
    # không mô hình nào dự đoán được gì, và so sánh MSE/frac giữa chúng vô
    # nghĩa (chênh lệch MSE lúc đó chỉ là nhiễu fit, không phải tín hiệu).
    var_y = float(np.var(Y))
    r2 = lambda m: 1.0 - m / max(1e-12, var_y)
    r2_solo, r2_mf, r2_core = r2(mse_solo), r2(mse_mf), r2(mse_core)

    print(f"""
TẦNG 1 — KHOẢNG CÁCH DỰ ĐOÁN GIÁ TRỊ

  R² solo={r2_solo:.3f}  mf={r2_mf:.3f}  core={r2_core:.3f}
  (đọc dòng này TRƯỚC: nếu cả ba ≈ 0, không mô hình nào dự đoán được gì --
   MSE/gain/frac bên dưới không đáng tin cậy để so sánh.)

  MSE chỉ obs riêng          = {mse_solo:.5f}
  MSE + mean-field (matched) = {mse_mf:.5f}
  MSE + core đích danh       = {mse_core:.5f}

  thông tin do biết core     = {gain_core:.5f}
  phần trong tổng thông tin  = {frac:.1%} (NaN nếu |gain_total| <= 1e-9)

  frac < 10%  -> mean-field đã bắt gần hết; biết core thêm được ít.
  frac > 30%  -> biết đích danh core mang thông tin đáng kể.
""")
    return {"mse_solo": mse_solo, "mse_mf": mse_mf, "mse_core": mse_core,
            "r2_solo": r2_solo, "r2_mf": r2_mf, "r2_core": r2_core,
            "frac_from_core": float(frac)}

if __name__ == "__main__":
    # Import môi trường của bạn
    # (Nếu file này nằm cùng thư mục envs, import trực tiếp như sau)
    try:
        from omni_arena import OmniArena
    except ImportError:
        from envs.omni_arena import OmniArena

    # Khởi tạo môi trường. 
    # Bật cờ conditional_gates=True để môi trường có sự phụ thuộc cấu trúc rõ rệt.
    env = OmniArena(
        enable_conditional_gates=True,
        enable_latency_ladder=False,
        enable_congestion=False,
        enable_structural_shift=False
    )

    print("======================================================================")
    print("ĐANG CHẠY ĐÁNH GIÁ TẦNG 1: KHOẢNG CÁCH DỰ ĐOÁN GIÁ TRỊ (PREDICTIVE)")
    print("======================================================================")
    print("Đang thu thập dữ liệu (Rollout)... Quá trình này có thể mất chút thời gian.")

    # Chạy Tier-1
    # Lưu ý: Nếu chạy 40 episodes mất quá nhiều thời gian, bạn có thể 
    # giảm n_episodes xuống 10 hoặc 20 để test code trước.
    results = run_tier1(
        env=env,
        n_episodes=40,    
        horizon=8,        
        discount=0.95,    
        k_core=2,         # Chọn 2 agent tác động mạnh nhất làm 'core'
        seed=123
    )