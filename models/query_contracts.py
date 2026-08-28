"""Typed, fail-closed query/oracle contracts for P01/P03/P08 experiments.

The same relation scores must not silently serve response, policy-contrast,
information, removal, and coalition queries.  This module makes the query,
estimand, support, budget, generator, and oracle provenance explicit before an
allocation experiment is allowed to rank candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Callable, Dict, Mapping, Tuple


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class QueryKind(str, Enum):
    RESPONSE_RANGE = "response_range"
    POLICY_CONTRAST = "policy_contrast"
    INFORMATION = "information"
    REMOVAL = "removal"
    COALITION = "coalition"


@dataclass(frozen=True)
class QueryCard:
    query_id: str
    kind: QueryKind
    estimand_key: str
    support_key: str
    candidate_ids: Tuple[str, ...]
    budget: int
    generator_sha256: str
    oracle_source_sha256: str
    cost_key: str = "unit_cost"
    tie_rule: str = "score_desc_then_candidate_id"
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("query_id", "estimand_key", "support_key", "cost_key"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.kind, QueryKind):
            raise TypeError("kind must be a QueryKind")
        if not isinstance(self.candidate_ids, tuple) or not self.candidate_ids:
            raise ValueError("candidate_ids must be non-empty")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if any(not str(candidate).strip() for candidate in self.candidate_ids):
            raise ValueError("candidate_ids must not contain empty identifiers")
        if type(self.budget) is not int:
            raise TypeError("budget must be an exact integer")
        if not 0 < self.budget <= len(self.candidate_ids):
            raise ValueError("budget must be in [1, number of candidates]")
        if self.tie_rule != "score_desc_then_candidate_id":
            raise ValueError("unsupported or nondeterministic tie_rule")
        for name in ("generator_sha256", "oracle_source_sha256"):
            if not _SHA256.fullmatch(str(getattr(self, name))):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported query-card schema version")

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OracleEvaluation:
    query_fingerprint: str
    scores: Mapping[str, float]
    selected: Tuple[str, ...]


class QueryOracleRegistry:
    """Registry whose evaluators are typed by an immutable ``QueryCard``."""

    def __init__(self) -> None:
        self._entries: Dict[str, Tuple[QueryCard, Callable[[], Mapping[str, float]]]] = {}

    def register(
        self, card: QueryCard, evaluator: Callable[[], Mapping[str, float]]
    ) -> str:
        if not callable(evaluator):
            raise TypeError("oracle evaluator must be callable")
        fingerprint = card.fingerprint()
        if card.query_id in self._entries:
            raise ValueError(f"duplicate query_id: {card.query_id}")
        self._entries[card.query_id] = (card, evaluator)
        return fingerprint

    def evaluate(self, query_id: str) -> OracleEvaluation:
        if query_id not in self._entries:
            raise KeyError(f"unregistered query_id: {query_id}")
        card, evaluator = self._entries[query_id]
        raw = dict(evaluator())
        expected = set(card.candidate_ids)
        if set(raw) != expected:
            missing = sorted(expected - set(raw))
            extra = sorted(set(raw) - expected)
            raise ValueError(f"oracle candidate mismatch: missing={missing}, extra={extra}")
        scores = {candidate: float(raw[candidate]) for candidate in card.candidate_ids}
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError("oracle returned a non-finite score")
        selected = tuple(sorted(
            card.candidate_ids,
            key=lambda candidate: (-scores[candidate], str(candidate)),
        )[: card.budget])
        return OracleEvaluation(
            query_fingerprint=card.fingerprint(), scores=scores, selected=selected
        )

    def require_kinds(self, *kinds: QueryKind) -> None:
        available = {card.kind for card, _ in self._entries.values()}
        missing = [kind.value for kind in kinds if kind not in available]
        if missing:
            raise RuntimeError(f"required query kinds are not registered: {missing}")
