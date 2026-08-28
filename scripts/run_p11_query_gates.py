"""Deterministic typed-query gates for the Paper-01/Paper-03 scope.

This runner is deliberately finite and synthetic.  It verifies that query
cards are executable, that response scores cannot impersonate information or
removal scores, and that the coupled-support/XOR witnesses are reproduced.
Passing is permission to run later allocation experiments, not evidence that a
learned MARL method works or that the paper is novel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from models.query_contracts import QueryCard, QueryKind, QueryOracleRegistry
from scripts.scientific_gate_common import atomic_json, gate_record


PROTOCOL_VERSION = "p11_typed_query_exact_gates_v1"
CANDIDATES = ("r0", "r1")


def _hash_json(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _card(query_id, kind, scores, generator):
    return QueryCard(
        query_id=query_id,
        kind=kind,
        estimand_key=f"finite-world-v1/{kind.value}",
        support_key="finite-two-relation-common-support",
        candidate_ids=CANDIDATES,
        budget=1,
        generator_sha256=_hash_json(generator),
        oracle_source_sha256=_hash_json({"query": query_id, "scores": scores}),
    )


def _select(scores):
    return sorted(scores, key=lambda candidate: (-float(scores[candidate]), candidate))[0]


def run_exact_query_gates():
    generator = {
        "candidate_ids": list(CANDIDATES),
        "budget": 1,
        "response": {"r0": 4.0, "r1": 1.0},
        "information": {"r0": 0.0, "r1": 3.0},
        "removal": {"r0": 2.0, "r1": 5.0},
    }
    score_tables = {
        QueryKind.RESPONSE_RANGE: generator["response"],
        QueryKind.INFORMATION: generator["information"],
        QueryKind.REMOVAL: generator["removal"],
    }
    registry = QueryOracleRegistry()
    query_ids = {}
    for kind, scores in score_tables.items():
        query_id = kind.value
        query_ids[kind] = query_id
        registry.register(
            _card(query_id, kind, scores, generator),
            lambda values=dict(scores): dict(values),
        )
    registry.require_kinds(*score_tables)
    evaluations = {
        kind.value: registry.evaluate(query_ids[kind]) for kind in score_tables
    }

    transfer_rows = []
    for selector_kind, selector_scores in score_tables.items():
        selected = _select(selector_scores)
        for target_kind, target_scores in score_tables.items():
            optimum = max(float(value) for value in target_scores.values())
            regret = optimum - float(target_scores[selected])
            transfer_rows.append({
                "selector": selector_kind.value,
                "target_query": target_kind.value,
                "selected": selected,
                "regret": regret,
            })
    matched = [
        row for row in transfer_rows if row["selector"] == row["target_query"]
    ]
    mismatched = [
        row for row in transfer_rows if row["selector"] != row["target_query"]
    ]

    # Same complete response scores, opposite information winners.
    same_response_worlds = {
        "world_a": {
            "response": {"r0": 0.0, "r1": 0.0},
            "information": {"r0": 1.0, "r1": 0.0},
        },
        "world_b": {
            "response": {"r0": 0.0, "r1": 0.0},
            "information": {"r0": 0.0, "r1": 1.0},
        },
    }
    same_response = (
        same_response_worlds["world_a"]["response"]
        == same_response_worlds["world_b"]["response"]
    )
    opposite_information = (
        _select(same_response_worlds["world_a"]["information"])
        != _select(same_response_worlds["world_b"]["information"])
    )

    # Exact coupled-support witness from the compiled Lean suite.
    singleton_radii = {"keep_1": 0.0, "keep_2": 5.5, "keep_3": 5.5}
    pair_radii = {"keep_12": 5.0, "keep_13": 5.0, "keep_23": 0.5}
    singleton_optimum = min(singleton_radii, key=singleton_radii.get)
    pair_optimum = min(pair_radii, key=pair_radii.get)
    singleton_set = {"keep_1": {1}, "keep_2": {2}, "keep_3": {3}}[
        singleton_optimum
    ]
    pair_set = {
        "keep_12": {1, 2}, "keep_13": {1, 3}, "keep_23": {2, 3}
    }[pair_optimum]
    nonnested = not singleton_set.issubset(pair_set)

    # XOR: zero single-coordinate marginal contrast, positive joint variation.
    xor_values = {(False, False): 0.0, (False, True): 1.0,
                  (True, False): 1.0, (True, True): 0.0}
    marginal_0 = 0.5 * (
        xor_values[(False, False)] + xor_values[(False, True)]
    ) - 0.5 * (
        xor_values[(True, False)] + xor_values[(True, True)]
    )
    marginal_1 = 0.5 * (
        xor_values[(False, False)] + xor_values[(True, False)]
    ) - 0.5 * (
        xor_values[(False, True)] + xor_values[(True, True)]
    )
    xor_pairwise_zero_joint_nonconstant = (
        marginal_0 == 0.0 and marginal_1 == 0.0
        and len(set(xor_values.values())) > 1
    )

    gates = [
        gate_record(
            "Q0", len(evaluations) == 3,
            required=True,
            metrics={
                kind: {
                    "query_fingerprint": row.query_fingerprint,
                    "selected": list(row.selected),
                    "scores": dict(row.scores),
                }
                for kind, row in evaluations.items()
            },
            rule="all typed response/information/removal cards execute with exact candidate coverage",
            failure_action="fix query registry/cards before any allocation experiment",
        ),
        gate_record(
            "Q1", same_response and opposite_information,
            required=True,
            metrics={
                "same_response": same_response,
                "opposite_information_winners": opposite_information,
                "worlds": same_response_worlds,
            },
            rule="two worlds with identical response scores must admit opposite information winners",
            failure_action="remove the response-not-sufficient-for-information claim",
        ),
        gate_record(
            "Q2",
            all(row["regret"] == 0.0 for row in matched)
            and any(row["regret"] > 0.0 for row in mismatched),
            required=True,
            metrics={"transfer_rows": transfer_rows},
            rule="matched selectors have zero own-query regret and at least one mismatched selector fails",
            failure_action="do not run learned query-matched allocation",
        ),
        gate_record(
            "Q3", nonnested,
            required=True,
            metrics={
                "singleton_radii": singleton_radii,
                "pair_radii": pair_radii,
                "singleton_optimum": sorted(singleton_set),
                "pair_optimum": sorted(pair_set),
            },
            rule="exact budget-one and budget-two support optima are nonnested",
            failure_action="remove the fixed-ranking impossibility witness",
        ),
        gate_record(
            "Q4", xor_pairwise_zero_joint_nonconstant,
            required=True,
            metrics={
                "marginal_0": marginal_0,
                "marginal_1": marginal_1,
                "joint_values": {str(key): value for key, value in xor_values.items()},
            },
            rule="XOR has zero pairwise marginal contrasts but nonconstant joint value",
            failure_action="remove the pairwise-not-sufficient-for-synergy witness",
        ),
    ]
    passed = all(row["passed"] for row in gates if row["required"])
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_class": "DETERMINISTIC_SYNTHETIC_GATE_ONLY",
        "overall_status": "PASS" if passed else "FAIL",
        "gate_pass": {row["gate"]: row["passed"] for row in gates},
        "gates": {row["gate"]: row for row in gates},
        "generator_sha256": _hash_json(generator),
        "limitations": [
            "no learned MARL method is evaluated",
            "no external method adapter is evaluated",
            "no novelty or empirical effect is established",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = run_exact_query_gates()
    atomic_json(Path(args.out), payload)
    print(json.dumps({
        "overall_status": payload["overall_status"],
        "gate_pass": payload["gate_pass"],
        "out": str(Path(args.out).resolve()),
    }, indent=2))
    raise SystemExit(0 if payload["overall_status"] == "PASS" else 2)


if __name__ == "__main__":
    main()

