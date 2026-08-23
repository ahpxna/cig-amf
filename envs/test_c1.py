import numpy as np
try:
    from envs.omni_arena import OmniArena
except ModuleNotFoundError:  # Allow direct execution from envs/.
    from omni_arena import OmniArena

env = OmniArena(n_agents=24, grid_size=24, n_zones=4)
env.reset()
z = 0
for name, pos_key in [("gate", "zone_gate"), ("resource", "zone_resource"), ("sink", "zone_sink")]:
    env.positions[0] = list(getattr(env, pos_key)[z])
    s, d = env._frenet_sd(0)
    print(f"{name}: s={s:.3f} d={d:.3f}")  # expectation: s ≈ 0 / ~0.375 / 1.0, d ≈ 0 for all three
