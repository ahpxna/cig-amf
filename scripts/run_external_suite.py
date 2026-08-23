"""Create a reproducible external-benchmark collection manifest.

This launcher only permits scientific panels that the selected adapter
explicitly supports.  It never substitutes a generic rollout for H1/H2/L.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from envs.external_contract import BenchmarkCapabilities


CAPABILITIES = {
    "flatland": BenchmarkCapabilities("Flatland", True, True, True, True, True, True, True),
    "rware": BenchmarkCapabilities("RWARE", True, True, False, False, True, True, False),
    "cyborg": BenchmarkCapabilities("CybORG", False, True, False, False, True, True, False),
    "cityflow": BenchmarkCapabilities("CityFlow", False, True, False, False, True, True, False),
}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=sorted(CAPABILITIES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panels", nargs="+", default=["training", "h1", "h2", "latency"])
    args = parser.parse_args(argv)
    cap = CAPABILITIES[args.environment]
    payload = {
        "environment": args.environment,
        "capabilities": vars(cap),
        "requested_panels": args.panels,
        "runnable_panels": [panel for panel in args.panels if cap.supports(panel)],
        "blocked_panels": [panel for panel in args.panels if not cap.supports(panel)],
        "rule": "blocked panels must not generate paper evidence",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
