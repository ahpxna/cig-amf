"""Fail-closed readiness adjudication for the currently selected paper scope.

Engineering readiness and scientific support are reported separately.  A
plumbing PASS never upgrades a scientific claim, and missing fields always
block the dependent experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_p11_query_gates import PROTOCOL_VERSION as QUERY_PROTOCOL
from scripts.scientific_gate_common import atomic_json, load_json


PROTOCOL_VERSION = "p11_scope_readiness_v1"
PRECHECK_PROTOCOL = "scientific_prechecks_g0_g4_v4_allocation_fidelity"


def _all_true(mapping, keys):
    return isinstance(mapping, dict) and all(mapping.get(key) is True for key in keys)


def _d0_status(payload):
    if not isinstance(payload, dict):
        return "NOT_RUN", False
    candidates = [
        payload.get("d0_status"),
        payload.get("overall_status"),
        (payload.get("gate_result") or {}).get("status")
        if isinstance(payload.get("gate_result"), dict) else None,
        (payload.get("d0_gate") or {}).get("status")
        if isinstance(payload.get("d0_gate"), dict) else None,
    ]
    status = next((str(value) for value in candidates if value is not None), "UNKNOWN")
    execution_complete = bool(
        payload.get("collection_complete", False)
        or payload.get("complete", False)
        or payload.get("records")
        or payload.get("rows")
    )
    return status, execution_complete


def validate_scope(*, query, prechecks, h1, periphery, d0=None):
    query_payload = load_json(query, "P11 query gates")
    precheck_payload = load_json(prechecks, "scientific prechecks")
    h1_payload = load_json(h1, "H1 claim")
    periphery_payload = load_json(periphery, "periphery manifest")
    d0_payload = load_json(d0, "D0 run") if d0 else None

    query_ok = bool(
        query_payload.get("protocol_version") == QUERY_PROTOCOL
        and query_payload.get("overall_status") == "PASS"
        and _all_true(query_payload.get("gate_pass"), ("Q0", "Q1", "Q2", "Q3", "Q4"))
    )
    prechecks_execute = bool(
        precheck_payload.get("protocol_version") == PRECHECK_PROTOCOL
        and set((precheck_payload.get("gates") or {})) == {"G0", "G1", "G2", "G3", "G4"}
        and precheck_payload.get("overall_status") in {"SMOKE_ONLY", "PASS", "FAIL"}
    )
    prechecks_confirmed = bool(
        precheck_payload.get("protocol_mode") == "confirmatory"
        and precheck_payload.get("core_prechecks_pass") is True
        and precheck_payload.get("overall_status") == "PASS"
    )
    h1_supported = bool(h1_payload.get("h1_claim_gate_pass") is True)
    periphery_plumbing = bool(
        periphery_payload.get("experiment") == "paper_b_fixed_core_periphery"
        and periphery_payload.get("complete") is True
        and int(periphery_payload.get("summary_row_count", 0)) > 0
    )
    paper08_information_science = bool(
        periphery_payload.get("information_channel_gate_pass") is True
        and periphery_payload.get("same_q_opposite_information_gate_pass") is True
        and len(str(periphery_payload.get("information_oracle_sha256", ""))) == 64
    )
    d0_gate_status, d0_executes = _d0_status(d0_payload)
    # A D0 status is scientific support only when the artifact explicitly
    # identifies a validated CIG-AMF policy. The current OmniArena runner
    # uses a prespecified surrogate and is therefore execution evidence, not
    # support for the learned-policy paper claim.
    d0_manifest = d0_payload.get("manifest", {}) if isinstance(d0_payload, dict) else {}
    d0_supported = bool(
        d0_gate_status == "CONFIRMED"
        and d0_payload.get("evidence_class") == "LEARNED_CIG_AMF_D0"
        and d0_manifest.get("live_policy_scope") == "validated_cig_amf_policy"
    )
    paper3_theory = False  # No T0/T1/T2 compiled artifact is accepted in v1.

    engineering = {
        "typed_query_exact_runner": query_ok,
        "g0_g4_quick_runner_executes": prechecks_execute,
        "paper08_periphery_plumbing_executes": periphery_plumbing,
        "paper07_09_d0_collector_executes": d0_executes,
    }
    scientific = {
        "paper06_h1_confirmatory_supported": h1_supported,
        "paper08_information_channel_supported": paper08_information_science,
        "paper07_09_d0_supported": d0_supported,
        "paper07_09_t0_t1_t2_formalized": paper3_theory,
        "g0_g4_confirmatory_supported": prechecks_confirmed,
    }

    allowed = {
        "paper01_03_exact_query_rerun": query_ok,
        "paper01_03_quick_allocation_diagnostic": query_ok and prechecks_execute,
        "paper08_plumbing_smoke": query_ok and periphery_plumbing,
        "paper07_09_d0_quick": d0_executes,
    }
    blocked = {
        "paper01_03_confirmatory_allocation": not prechecks_confirmed,
        "paper06_confirmatory_claim": not h1_supported,
        "paper08_scientific_training_grid": not paper08_information_science,
        "paper07_09_adaptive_algorithm": not (d0_supported and paper3_theory),
    }
    engineering_count = sum(engineering.values())
    scientific_count = sum(scientific.values())
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "engineering_execution_contracts": engineering,
        "engineering_alignment": {
            "passed": engineering_count,
            "total": len(engineering),
            "fraction": engineering_count / len(engineering),
        },
        "scientific_support_contracts": scientific,
        "scientific_support": {
            "passed": scientific_count,
            "total": len(scientific),
            "fraction": scientific_count / len(scientific),
        },
        "allowed_now": allowed,
        "blocked_now": blocked,
        "paper_status": {
            "paper01_03": "QUICK_DIAGNOSTIC_READY" if allowed[
                "paper01_03_quick_allocation_diagnostic"
            ] else "BLOCKED",
            "paper06": "CONFIRMATORY_SUPPORTED" if h1_supported else "BLOCKED_H1",
            "paper08": (
                "SCIENTIFIC_GRID_READY" if paper08_information_science
                else "PLUMBING_ONLY_SCIENCE_BLOCKED"
            ),
            "paper07_09": (
                "ALGORITHM_READY" if d0_supported and paper3_theory
                else f"D0_{d0_gate_status}_THEORY_UNFORMALIZED"
            ),
        },
        "can_start_unbounded_or_confirmatory_grid": not any(blocked.values()),
        "evidence_boundary": (
            "engineering execution is separate from scientific support; smoke, "
            "synthetic, and plumbing results never count as claim confirmation"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--prechecks", required=True)
    parser.add_argument("--h1", required=True)
    parser.add_argument("--periphery", required=True)
    parser.add_argument("--d0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    payload = validate_scope(
        query=args.query,
        prechecks=args.prechecks,
        h1=args.h1,
        periphery=args.periphery,
        d0=args.d0,
    )
    atomic_json(Path(args.out), payload)
    print(json.dumps({
        "engineering_alignment": payload["engineering_alignment"],
        "scientific_support": payload["scientific_support"],
        "paper_status": payload["paper_status"],
        "can_start_unbounded_or_confirmatory_grid": payload[
            "can_start_unbounded_or_confirmatory_grid"
        ],
        "out": str(Path(args.out).resolve()),
    }, indent=2))
    return payload


if __name__ == "__main__":
    main()
