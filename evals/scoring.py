"""Deterministic field scoring. The harness owns this logic, so a pass means exactly what
these few functions say it means.

Extracting candidate values from the agent's answer is the harness's job (Phase 6);
these functions only compare values that have already been extracted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from evals.models import FieldSpec


def _norm(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").removesuffix("%")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def match_field(spec: FieldSpec, candidate: Any) -> bool:
    if candidate is None:
        return False
    if spec.match == "numeric":
        got = _to_float(candidate)
        if got is None or not math.isfinite(got):
            return False
        want = float(spec.value)  # type: ignore[arg-type]  # validated numeric
        rel, abs_ = spec.tolerance
        return math.isclose(got, want, rel_tol=rel, abs_tol=abs_)
    if spec.match == "exact":
        accepted = {_norm(spec.value), *(_norm(a) for a in spec.aliases)}
        return _norm(candidate) in accepted
    if not isinstance(candidate, list | tuple):
        return False
    want_items = [_norm(v) for v in spec.value]  # type: ignore[union-attr]  # validated list
    got_items = [_norm(v) for v in candidate]
    if spec.match == "set":
        return len(got_items) == len(set(got_items)) and set(got_items) == set(want_items)
    return got_items == want_items


@dataclass(frozen=True)
class FieldResult:
    name: str
    passed: bool
    expected: Any
    got: Any


def score_fields(
    expected: Mapping[str, FieldSpec], candidate: Mapping[str, Any]
) -> list[FieldResult]:
    return [
        FieldResult(name, match_field(spec, candidate.get(name)), spec.value, candidate.get(name))
        for name, spec in expected.items()
    ]


def all_pass(results: list[FieldResult]) -> bool:
    return bool(results) and all(r.passed for r in results)
