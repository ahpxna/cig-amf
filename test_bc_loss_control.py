"""Control experiment for BUG B (debug master, Section 3.4).

For |A|=6, a uniform predictor has cross-entropy ln(6)=1.7918. Recorded logs
placed bc_loss around 2.29-2.39, ABOVE that floor. This indicates confidently
wrong predictions, a classic sign of label alignment errors such as temporal
off-by-one or neighbor-index mismatch, rather than insufficient model capacity.

Two decisive tests are defined:

T1, deterministic scripted policy: every agent repeats
``a_j^t=(t+j)%A``, so ``a_j^{t+1}=(a_j^t+1)%A`` is a deterministic, trivial
function of ``a_j^t`` already present in x_full. A correctly aligned pipeline
must drive bc_loss toward zero. A value near 1.79 means the model cannot use the
input; a value near the recorded 2.3 confirms label misalignment.

T2, shuffled labels: target_action is randomly permuted in bc_buffer before
training. The target is independent of the input, so the original test expected
ln(6). [P-5] later established that reported bc_loss includes a 0.25-weighted
shadow loss, making the correct floor ``(1+0.25)*ln(6)=2.2397``. The earlier
2.2336 result matches that floor within 6e-3, showing that the pipeline was not
broken; the test's expected constant was wrong.

Gate: T1 < 0.3. T2 within 0.15 of ``(1+0.25)*ln(6)``.

Run: ``python3 test_bc_loss_control.py``. A real PyTorch installation is
required: either CPU PyTorch or a host with the runtime required by a CUDA-only
build.
"""
import random

import numpy as np
import torch

from envs.omni_arena import OmniArena
from models.core_behavior import PairRelationalModule


def _deterministic_actions(env, t):
    """Return deterministic ``a_j^t=(t+j)%A`` actions independent of RNG."""
    A = env.get_action_dim()
    return [int((t + j) % A) for j in range(env.n_agents)]


def collect_bc_buffer(env, pair_rel, n_steps, action_fn, seed=0):
    """Reproduce the snapshot timing used by ``collect_episode()``.

    ``obs_all`` and ``actions_list`` are context at t. ``step_population()``
    must observe geometry at t, not t+1, while ``add_bc_transition()`` pairs
    context at t with ``h_prev=z_ij^{t-1}`` and target ``a_j^{t+1}``.
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
    print("T1 — deterministic scripted policy: BC loss should converge to ~0")
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
    print(f"\n  final BC loss = {loss:.4f}  (uniform floor ln6 = {ln6:.4f})")

    ok = loss < 0.3
    print(f"  [{'PASS' if ok else 'FAIL'}] T1: bc_loss < 0.3  (gate G2)")

    if not ok and loss > ln6:
        print(
            "  >> BC loss is above the uniform floor on a deterministic task. "
            "This confirms a label-alignment defect rather than a learning-rate "
            "or model-capacity issue."
        )
    elif not ok:
        print(
            "  >> BC loss is below ln6 but not close to zero. More epochs or "
            "capacity may be needed; do not infer a label-alignment defect yet."
        )

    return ok


def test_t2_label_shuffle():
    print("\n" + "=" * 70)
    print("T2 — shuffled labels: BC loss should converge to the corrected floor")
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
    # [P-5 FINAL DEBUG] Reported bc_loss includes the shadow loss with
    # shadow_loss_weight=0.25. The shuffled-label floor is therefore
    # (1+0.25)*ln6=2.2397, not ln6=1.7918. The previous 2.2336 measurement
    # matches the correct floor within 6e-3: the pipeline was never broken;
    # the expected test constant was wrong.
    floor = (1.0 + 0.25) * ln6
    print(f"\n  final BC loss = {loss:.4f}  (correct floor (1+0.25)·ln6 = {floor:.4f})")

    ok = abs(loss - floor) < 0.15
    print(f"  [{'PASS' if ok else 'FAIL'}] T2: |bc_loss - (1+0.25)·ln6| < 0.15")

    if not ok:
        print(
            "  >> Labels are shuffled but BC loss differs from the "
            "(1+w_shadow)·ln6 floor. Check batching, loss composition, and "
            "shadow_loss_weight."
        )

    return ok


if __name__ == "__main__":
    ok1 = test_t1_scripted_policy_deterministic()
    ok2 = test_t2_label_shuffle()

    print("\n" + "=" * 70)
    print(f"RESULT: T1={'PASS' if ok1 else 'FAIL'}  T2={'PASS' if ok2 else 'FAIL'}")
    print("=" * 70)

    if not (ok1 and ok2):
        raise SystemExit(1)
