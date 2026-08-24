"""Single registry for pinned external repositories, factories and capabilities."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib
import json
import os
import sys

from envs.external.rware import RWARECIGEnvironment
from envs.external.cyborg import CybORGCIGEnvironment
from envs.external.cityflow import CityFlowCIGEnvironment
from envs.flatland_adapter import FlatlandCIGEnvironment

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = ROOT / "external_envs"
REPOS_DIRNAME = "repos"

@dataclass(frozen=True)
class ExternalSpec:
    key: str
    repo_dir: str
    module: str
    factory: str
    capabilities: object

SPECS = {
    "flatland": ExternalSpec("flatland", "flatland-rl", "envs.external.registry", "make_flatland", FlatlandCIGEnvironment.capabilities),
    "rware": ExternalSpec("rware", "robotic-warehouse", "envs.external.rware", "make_rware_environment", RWARECIGEnvironment.capabilities),
    "cyborg": ExternalSpec("cyborg", "CybORG", "envs.external.cyborg", "make_cyborg_environment", CybORGCIGEnvironment.capabilities),
    "cityflow": ExternalSpec("cityflow", "CityFlow", "envs.external.cityflow", "make_cityflow_environment", CityFlowCIGEnvironment.capabilities),
}

def external_root():
    root = Path(os.environ.get("CIG_EXTERNAL_ENVS_DIR", DEFAULT_EXTERNAL_ROOT)).resolve()
    # Accept either .../external_envs or .../external_envs/repos as the override
    # without accidentally creating repos/repos.
    return root.parent if root.name == REPOS_DIRNAME else root

def repo_path(key):
    spec = SPECS[str(key)]
    root = external_root()
    canonical = root / REPOS_DIRNAME / spec.repo_dir
    legacy = root / spec.repo_dir
    return canonical if canonical.exists() or not legacy.exists() else legacy

def ensure_repo_on_path(key):
    path = repo_path(key)
    if not path.exists():
        raise FileNotFoundError(f"external repository not installed: {path}; run scripts/setup_external_envs.sh")
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return path

def build_environment(key, **kwargs):
    key = str(key)
    spec = SPECS[key]
    repo = ensure_repo_on_path(key)
    module = importlib.import_module(spec.module)
    factory = getattr(module, spec.factory)
    return factory(repo_path=repo, **kwargs)

def make_flatland(seed=0, n_agents=6, observation_width=256, max_steps=60, **_):
    del max_steps
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.rail_generators import sparse_rail_generator
    from flatland.envs.line_generators import sparse_line_generator
    rail = RailEnv(
        width=25, height=25,
        rail_generator=sparse_rail_generator(
            max_num_cities=3, seed=int(seed), grid_mode=False
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=int(n_agents), random_seed=int(seed),
    )
    env = FlatlandCIGEnvironment(rail, observation_width=observation_width)
    env.reset(seed=int(seed))
    return env

def registry_payload():
    return {
        key: {
            "repo": str(repo_path(key)),
            "exists": repo_path(key).exists(),
            "capabilities": vars(spec.capabilities),
        }
        for key, spec in SPECS.items()
    }
