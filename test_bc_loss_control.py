"""
test_bc_loss_control.py — Control experiment cho BUG B (xem
docs/CIG-AMF_training_debug_master.md mục 3.4).

|A| = 6 -> cross-entropy của một predictor đoán uniform là ln(6) = 1.7918.
Log thật cho thấy bc_loss đứng ở ~2.29-2.39, CAO HƠN sàn uniform -- tức là
predictor đang TỰ TIN DỰ ĐOÁN SAI, dấu hiệu kinh điển của lỗi căn chỉnh
nhãn (off-by-one thời gian, lệch chỉ số neighbour, ...), không phải lỗi
dung lượng mô hình.

Hai test quyết định:

  T1 (scripted policy XÁC ĐỊNH): mỗi agent lặp một hành động cố định
     a_j^t = (t + j) % A, nên a_j^{t+1} = (a_j^t + 1) % A -- một hàm
     TẤT ĐỊNH và TẦM THƯỜNG của chính input a_j^t đã có sẵn trong x_full.
     Nếu pipeline căn chỉnh đúng, bc_loss PHẢI hội tụ về ~0.
     Nếu nó đứng ở ~1.79 -> model không nhìn thấy input hữu ích.
     Nếu nó đứng ở ~2.3 (đúng như log thật) -> XÁC NHẬN nhãn bị lệch.

  T2 (label shuffle): xáo trộn target_action ngẫu nhiên trong bc_buffer
     trước khi train. Nhãn lúc này độc lập hoàn toàn với input, nên
     bc_loss phải hội tụ về ĐÚNG ln(6) = 1.7918 (không hơn, không kém).
     Nếu nó lại lệch khỏi 1.79 (đặc biệt nếu > 1.79) thì bản thân pipeline
     nhãn/batching đã hỏng, độc lập với việc nhãn có ý nghĩa hay không.

Gate (mục 3.5 trong doc): T1 phải < 0.3. T2 phải xấp xỉ 1.79 (+-0.1).

Chạy: python3 test_bc_loss_control.py
Cần torch thật (không chạy được trong sandbox không có CUDA runtime nếu
bản torch cài là bản cuda-only -- cần torch CPU hoặc máy có CUDA thật).
"""
import random

import numpy as np
import torch

from envs.omni_arena import OmniArena
from models.core_behavior import PairRelationalModule


def _deterministic_actions(env, t):
    """a_j^t = (t + j) % A -- tất định, không phụ thuộc rng."""
    A = env.get_action_dim()
    return [int((t + j) % A) for j in range(env.n_agents)]


def collect_bc_buffer(env, pair_rel, n_steps, action_fn, seed=0):
    """
    Chạy đúng loại timing snapshot mà final_runner.py::collect_episode()
    dùng (xem docstring ở đó): obs_all/actions_list là context tại t,
    step_population() phải thấy geometry tại t (không phải t+1), và
    add_bc_transition() phải ghép context tại t với h_prev = z_ij^{t-1}
    và target = a_j^{t+1}.
    """
    obs_all = env.reset()

    prev_obs_all = None
    prev_actions = None
    prev_env_snapshot_before_step = None
    prev_h_snapshot = None

    for t in range(int(n_steps)):
        actions_list = action_fn(env, t)

        env_snapshot_before_step = env.clone_state()
        h_snapshot_before_latent_update = pair_rel.clone_full_states_np()

        next_obs_all, rewards, done, info = env.step(actions_list)
        env_snapshot_after_step = env.clone_state()

        if (
            prev_obs_all is not None
            and prev_actions is not None
            and prev_env_snapshot_before_step is not None
        ):
            env.restore_state(prev_env_snapshot_before_step)

            pair_rel.add_bc_transition(
                observations={a: prev_obs_all[a] for a in range(env.n_agents)},
                actions={a: prev_actions[a] for a in range(env.n_agents)},
                next_actions={a: actions_list[a] for a in range(env.n_agents)},
                env=env,
                h_prev_snapshot=prev_h_snapshot,
            )

            env.restore_state(env_snapshot_after_step)

        env.restore_state(env_snapshot_before_step)

        pair_rel.step_population(obs_all=obs_all, actions=actions_list, env=env)

        env.restore_state(env_snapshot_after_step)

        prev_obs_all = [x.copy() for x in obs_all]
        prev_actions = list(actions_list)
        prev_env_snapshot_before_step = env_snapshot_before_step
        prev_h_snapshot = h_snapshot_before_latent_update

        obs_all = next_obs_all

        if done:
            obs_all = env.reset()
            prev_obs_all = None
            prev_actions = None
            prev_env_snapshot_before_step = None
            prev_h_snapshot = None


def _make_env_and_module(seed=0):
    env = OmniArena(
        n_agents=20, grid_size=12, n_zones=4,
        max_steps=200, phase_length=1000,
        enable_conditional_gates=False,
        enable_latency_ladder=False,
        enable_congestion=False,
        enable_structural_shift=False,
        seed=seed,
    )
    pair_rel = PairRelationalModule(
        n_agents=env.n_agents,
        obs_dim=env.get_obs_dim(),
        action_dim=env.get_action_dim(),
        hidden_dim=32,
        shadow_dim=16,
        rel_feat_dim=6,
        lr=1e-3,
        bc_buffer_size=200000,
        grad_clip=1.0,
        shadow_loss_weight=0.25,
        device="cpu",
    )
    return env, pair_rel


def test_t1_scripted_policy_deterministic():
    print("\n" + "=" * 70)
    print("T1 — scripted policy XÁC ĐỊNH: bc_loss phải hội tụ về ~0")
    print("=" * 70)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    env, pair_rel = _make_env_and_module(seed=0)
    collect_bc_buffer(env, pair_rel, n_steps=150, action_fn=_deterministic_actions)

    print(f"  bc_buffer size = {len(pair_rel.bc_buffer)}")

    loss = None
    for epoch in range(60):
        loss = pair_rel.train_bc(n_steps=20, batch_size=512)
        if epoch % 10 == 0 or epoch == 59:
            print(f"  epoch {epoch:3d}  bc_loss = {loss:.4f}")

    ln6 = float(np.log(6))
    print(f"\n  bc_loss cuối cùng = {loss:.4f}  (sàn uniform ln6 = {ln6:.4f})")

    ok = loss < 0.3
    print(f"  [{'PASS' if ok else 'FAIL'}] T1: bc_loss < 0.3  (gate G2)")

    if not ok and loss > ln6:
        print(
            "  >> bc_loss NẰM TRÊN sàn uniform trên bài toán TẤT ĐỊNH TUYỆT ĐỐI "
            "-> XÁC NHẬN bug căn chỉnh nhãn (mục 3.3 trong debug doc). "
            "Không phải learning rate, không phải dung lượng model."
        )
    elif not ok:
        print(
            "  >> bc_loss dưới ln6 nhưng chưa về gần 0 -> có thể chỉ là thiếu "
            "epoch/dung lượng, không nhất thiết là bug căn chỉnh. Tăng epoch/"
            "batch rồi thử lại trước khi kết luận."
        )

    return ok


def test_t2_label_shuffle():
    print("\n" + "=" * 70)
    print("T2 — label shuffle: bc_loss phải hội tụ về ĐÚNG ln(6) = 1.7918")
    print("=" * 70)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    env, pair_rel = _make_env_and_module(seed=0)
    collect_bc_buffer(env, pair_rel, n_steps=150, action_fn=_deterministic_actions)

    rng = np.random.RandomState(0)
    A = env.get_action_dim()
    for sample in pair_rel.bc_buffer:
        sample["target_action"] = int(rng.randint(0, A))

    loss = None
    for epoch in range(60):
        loss = pair_rel.train_bc(n_steps=20, batch_size=512)
        if epoch % 10 == 0 or epoch == 59:
            print(f"  epoch {epoch:3d}  bc_loss = {loss:.4f}")

    ln6 = float(np.log(6))
    # [P-5 FINAL DEBUG] bc_loss báo cáo GỘP shadow loss (shadow_loss_weight=0.25
    # ở khởi tạo module phía trên). Sàn đúng khi shuffle nhãn do đó là
    # (1 + 0.25) * ln6 = 2.2397, KHÔNG phải ln6 = 1.7918. Lần chạy trước đo
    # được 2.2336 — khớp sàn đúng tới 6e-3: pipeline chưa bao giờ hỏng, chỉ
    # hằng số kỳ vọng của test sai.
    floor = (1.0 + 0.25) * ln6
    print(f"\n  bc_loss cuối cùng = {loss:.4f}  (sàn đúng (1+0.25)·ln6 = {floor:.4f})")

    ok = abs(loss - floor) < 0.15
    print(f"  [{'PASS' if ok else 'FAIL'}] T2: |bc_loss - (1+0.25)·ln6| < 0.15")

    if not ok:
        print(
            "  >> Nhãn shuffle thật nhưng bc_loss lệch khỏi sàn (1+w_shadow)·ln6 "
            "-> kiểm tra lại pipeline batching/loss hoặc shadow_loss_weight."
        )

    return ok


if __name__ == "__main__":
    ok1 = test_t1_scripted_policy_deterministic()
    ok2 = test_t2_label_shuffle()

    print("\n" + "=" * 70)
    print(f"KẾT QUẢ: T1={'PASS' if ok1 else 'FAIL'}  T2={'PASS' if ok2 else 'FAIL'}")
    print("=" * 70)

    if not (ok1 and ok2):
        raise SystemExit(1)
