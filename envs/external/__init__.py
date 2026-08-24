"""External benchmark adapters.

Keep package import side-effect free: optional benchmark modules and the
registry are loaded lazily so importing one adapter never forces every other
optional dependency or creates a Flatland/registry cycle.
"""

__all__ = ["SPECS", "build_environment", "external_root", "repo_path"]


def __getattr__(name):
    if name in __all__:
        from envs.external import registry
        return getattr(registry, name)
    raise AttributeError(name)
