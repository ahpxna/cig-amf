"""
test_p2_p4_guards.py — kiểm chứng hai guard chống bẫy P2<->P4 (xem báo cáo
triển khai staged-audit): structural shift chỉ được trigger tại ranh giới
episode, và oracle rollout không được bắt đầu quá muộn để cửa sổ H bước
chạm ranh giới đó. Cả hai PHẢI raise AssertionError khi bị vi phạm, không
được âm thầm chạy qua.

Chạy: python3 test_p2_p4_guards.py
"""
from omni_arena import OmniArena
from tiny_oracle_dig import TinyOracleDIG


def test_guard_a_mid_episode_shift_asserts():
    env = OmniArena(
        n_agents=24, n_zones=4, max_steps=60, phase_length=6,
        causal_horizon=8, seed=1, mode="structural_shift",
    )
    env.reset()
    acts = [env.scripted_policy(i) for i in range(env.n_agents)]
    env.step(acts)  # t=1, done=False -- mid episode now
    try:
        env._do_structural_shift()
    except AssertionError:
        return True
    raise RuntimeError("GUARD A did not fire on a mid-episode structural shift attempt")


def test_guard_a_boundary_shift_does_not_assert():
    env = OmniArena(
        n_agents=24, n_zones=4, max_steps=6, phase_length=1,
        causal_horizon=8, seed=1, mode="structural_shift",
    )
    env.reset()
    for _ in range(6):
        acts = [env.scripted_policy(i) for i in range(env.n_agents)]
        env.step(acts)
    # Now done=True (episode finished) -- reset() should be allowed to shift.
    env.reset()
    return True


def test_guard_b_rollout_crossing_boundary_asserts():
    env = OmniArena(
        n_agents=24, n_zones=4, max_steps=20, phase_length=6,
        causal_horizon=8, seed=1,
    )
    env.reset()
    for _ in range(15):
        acts = [env.scripted_policy(i) for i in range(env.n_agents)]
        _, _, done, _ = env.step(acts)
        if done:
            break
    assert env.t + 8 > env.max_steps, "test setup invalid: t+horizon should exceed max_steps"
    try:
        env.rollout_from_current_state(forced=(0, env.STAY, 0), horizon=8)
    except AssertionError:
        pass
    else:
        raise RuntimeError("GUARD B did not fire on rollout_from_current_state")

    try:
        env.compute_oracle_influence_from_current_state(
            ego_id=0, agent_j=1, intervention_action=env.STAY,
            horizon=8, forced_step=0,
        )
    except AssertionError:
        return True
    raise RuntimeError("GUARD B did not fire via compute_oracle_influence_from_current_state")


def test_guard_b_safe_rollout_does_not_assert():
    env = OmniArena(
        n_agents=24, n_zones=4, max_steps=60, phase_length=6,
        causal_horizon=8, seed=1,
    )
    env.reset()
    env.rollout_from_current_state(forced=(0, env.STAY, 0), horizon=8)
    return True


def test_guard_b_applies_to_tiny_oracle_dig_too():
    env = TinyOracleDIG(max_steps=20, causal_horizon=8, seed=1)
    env.reset()
    for _ in range(15):
        acts = [env.scripted_policy(i) for i in range(env.n_agents)]
        _, _, done, _ = env.step(acts)
        if done:
            break
    try:
        env.rollout_from_current_state(forced=(0, env.STAY, 0), horizon=8)
    except AssertionError:
        return True
    raise RuntimeError("GUARD B did not fire on TinyOracleDIG.rollout_from_current_state")


def main():
    tests = [
        test_guard_a_mid_episode_shift_asserts,
        test_guard_a_boundary_shift_does_not_assert,
        test_guard_b_rollout_crossing_boundary_asserts,
        test_guard_b_safe_rollout_does_not_assert,
        test_guard_b_applies_to_tiny_oracle_dig_too,
    ]
    n_ok = 0
    for t in tests:
        try:
            t()
            print(f"[OK]   {t.__name__}")
            n_ok += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{n_ok}/{len(tests)} guard tests passed.")
    if n_ok != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
