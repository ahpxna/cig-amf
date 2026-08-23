"""Create a reproducible external-benchmark collection manifest.

This launcher only permits scientific panels that the selected adapter
explicitly supports.  It never substitutes a generic rollout for H1/H2/L.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.external_contract import BenchmarkCapabilities
from envs.flatland_adapter import FlatlandCIGEnvironment


CAPABILITIES = {
    # A wrapper owns its capability declaration. Do not duplicate booleans in
    # this collection launcher: a stale table can silently authorize claims
    # whose intervention/oracle methods do not exist.
    "flatland": FlatlandCIGEnvironment.capabilities,
    # No runnable CIG-AMF adapter exists for these environments yet. Keeping
    # them explicit and closed prevents a manifest from implying H1/H2/L
    # support just because an upstream simulator can be installed.
    "rware": BenchmarkCapabilities("RWARE", False, False, False, False, False, False, False),
    "cyborg": BenchmarkCapabilities("CybORG", False, False, False, False, False, False, False),
    "cityflow": BenchmarkCapabilities("CityFlow", False, False, False, False, False, False, False),
}


def _panel_support(environment, panel):
    cap = CAPABILITIES[environment]
    if not cap.supports(panel):
        return False, "capability flag is disabled"
    if environment == "flatland":
        # Validate the executable contract, not only declarative flags.
        # Methods are inspected without constructing a simulator instance.
        required = {
            "h1": ("clone_state", "restore_state", "fixed_continuation_policy", "oracle_lag_response"),
            "h2": ("clone_state", "restore_state", "apply_structural_intervention", "apply_behavioural_intervention"),
            "latency": ("clone_state", "restore_state", "fixed_continuation_policy", "oracle_lag_response"),
        }.get(panel, ())
        missing = [name for name in required if not callable(getattr(FlatlandCIGEnvironment, name, None))]
        if missing:
            return False, "missing executable methods: " + ", ".join(missing)
    return True, ""


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=sorted(CAPABILITIES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panels", nargs="+", default=["training", "h1", "h2", "latency"])
    args = parser.parse_args(argv)
    cap = CAPABILITIES[args.environment]
    decisions = {panel: _panel_support(args.environment, panel) for panel in args.panels}
    payload = {
        "environment": args.environment,
        "capabilities": vars(cap),
        "requested_panels": args.panels,
        "runnable_panels": [panel for panel, (ok, _) in decisions.items() if ok],
        "blocked_panels": [panel for panel, (ok, _) in decisions.items() if not ok],
        "blocked_reasons": {
            panel: reason for panel, (ok, reason) in decisions.items() if not ok
        },
        "rule": "blocked panels must not generate paper evidence",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
