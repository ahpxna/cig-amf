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
training. The target is independent of the input.  The two-tier architecture
has two distinct valid floors: shadow-only records contribute
``0.25*ln(6)``, while records whose pair is explicitly core-active contribute
``(1+0.25)*ln(6)``.  Both controls are checked so an allocation regression
cannot masquerade as a label-alignment result.

Gate: T1 < 0.3. T2a within 0.15 of ``0.25*ln(6)``. T2b within 0.15 of
``(1+0.25)*ln(6)``.

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
        n_agents=8, grid_size=13, n_zones=1,
        max_steps=64, phase_length=1000,
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
        bc_buffer_size=12000,
        grad_clip=1.0,
        shadow_loss_weight=0.25,
        device="cpu",
    )
    return env, pair_rel


def _activate_all_pairs(env, pair_rel):
    """Seed the full recurrent tier explicitly for core-active controls."""
    pair_rel.reconcile_core_sets({
        ego: [neighbor for neighbor in range(env.n_agents) if neighbor != ego]
        for ego in range(env.n_agents)
    })


def test_t1_scripted_policy_deterministic():
    print("\n" + "=" * 70)
    print("T1 — deterministic scripted policy: BC loss should converge to ~0")
    print("=" * 70)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    env, pair_rel = _make_env_and_module(seed=0)
    _activate_all_pairs(env, pair_rel)
    collect_bc_buffer(env, pair_rel, n_steps=48, action_fn=_deterministic_actions)

    print(f"  bc_buffer size = {len(pair_rel.bc_buffer)}")

    loss = None
    for epoch in range(24):
        loss = pair_rel.train_bc(n_steps=4, batch_size=128)
        if epoch % 6 == 0 or epoch == 23:
            print(f"  epoch {epoch:3d}  bc_loss = {loss:.4f}")
        if loss < 0.2:
            break

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


def _shuffle_and_fit(pair_rel, action_dim):
    rng = np.random.RandomState(0)
    for sample in pair_rel.bc_buffer:
        sample["target_action"] = int(rng.randint(0, action_dim))
    loss = None
    # Shuffled-label floors are analytic. A short optimization only verifies
    # that the implementation reaches the expected allocation-weighted loss.
    for _ in range(16):
        loss = pair_rel.train_bc(n_steps=3, batch_size=128)
    return float(loss)


def test_t2_label_shuffle():
    print("\n" + "=" * 70)
    print("T2 — shuffled labels: BC loss should converge to the corrected floor")
    print("=" * 70)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    shadow_env, shadow_pair_rel = _make_env_and_module(seed=0)
    collect_bc_buffer(
        shadow_env, shadow_pair_rel, n_steps=48,
        action_fn=_deterministic_actions,
    )
    core_env, core_pair_rel = _make_env_and_module(seed=0)
    _activate_all_pairs(core_env, core_pair_rel)
    collect_bc_buffer(
        core_env, core_pair_rel, n_steps=48,
        action_fn=_deterministic_actions,
    )
    A = shadow_env.get_action_dim()
    shadow_loss = _shuffle_and_fit(shadow_pair_rel, A)
    core_loss = _shuffle_and_fit(core_pair_rel, A)
    ln6 = float(np.log(6))
    shadow_floor = 0.25 * ln6
    core_floor = (1.0 + 0.25) * ln6
    print(f"\n  shadow-only loss = {shadow_loss:.4f}  (floor = {shadow_floor:.4f})")
    print(f"  core-active loss = {core_loss:.4f}  (floor = {core_floor:.4f})")
    ok_shadow = abs(shadow_loss - shadow_floor) < 0.15
    ok_core = abs(core_loss - core_floor) < 0.15
    ok = ok_shadow and ok_core
    print(f"  [{'PASS' if ok_shadow else 'FAIL'}] T2a: shadow-only floor")
    print(f"  [{'PASS' if ok_core else 'FAIL'}] T2b: core-active floor")

    if not ok:
        print(
            "  >> Shuffled-label loss differs from its state-allocation "
            "floor. Check core allocation, batch composition, and loss weights."
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
