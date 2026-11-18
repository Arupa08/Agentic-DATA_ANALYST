"""Phase 2 exit checks: every case has verified ground truth, every dirty-data trap is
real, the set matches the plan's shape, and the files are frozen."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest
from pydantic import ValidationError

from evals.models import EvalCase, EvalSet, load_cases
from evals.verify import CASES_PATH, HOLDOUT_PATH, check_lock, verify, verify_case


@pytest.fixture(scope="module")
def main_set() -> EvalSet:
    return load_cases(CASES_PATH)


@pytest.fixture(scope="module")
def holdout_set() -> EvalSet:
    return load_cases(HOLDOUT_PATH)


def test_all_ground_truth_verified(
    shop_db: sqlite3.Connection, main_set: EvalSet, holdout_set: EvalSet
) -> None:
    assert verify(shop_db, main_set, holdout_set) == []


def test_verifier_catches_wrong_answer_and_fake_trap(
    shop_db: sqlite3.Connection, main_set: EvalSet
) -> None:
    case = next(c for c in main_set.cases if c.id == "dirty_05")
    wrong = case.model_copy(
        update={"expected": {"revenue": case.expected["revenue"].model_copy(update={"value": 1.0})}}
    )
    assert any("stored" in p for p in verify_case(shop_db, wrong))
    fake_trap = case.model_copy(update={"naive_sql": case.sql})
    assert any("not a trap" in p for p in verify_case(shop_db, fake_trap))


def test_case_files_are_frozen() -> None:
    assert check_lock() == []


def test_every_flaw_is_trapped_somewhere(main_set: EvalSet) -> None:
    trapped = {t for c in main_set.cases for t in c.traps}
    # D3 (null timestamps) never changes a filtered answer; it is exercised via multi_05
    assert {"D1", "D2", "D4", "D5", "D6", "D7"} <= trapped
    assert "D3" in trapped


# ---------------------------------------------------------------- schema rejects bad cases

GOOD: dict[str, Any] = {
    "id": "simple_99",
    "category": "simple_lookup",
    "question": "How many orders?",
    "expected_behavior": "answer",
    "sql": "SELECT COUNT(*) AS n FROM orders",
    "expected": {"n": {"match": "numeric", "value": 1}},
}


def test_good_case_loads() -> None:
    EvalCase.model_validate(GOOD)


@pytest.mark.parametrize(
    "patch",
    [
        {"unexpected_key": 1},
        {"expected_behavior": "blocked"},
        {"sql": None},
        {"expected": {}},
        {"expected": {"n": {"match": "numeric", "value": "ten"}}},
        {"expected": {"n": {"match": "exact", "value": "x", "rel_tol": 0.1}}},
        {
            "expected": {
                "n": {"match": "set", "value": ["a"]},
                "m": {"match": "numeric", "value": 1},
            }
        },
        {"traps": ["D9"]},
        {"id": "Bad-ID"},
        {"category": "dirty_data"},  # dirty cases need naive_sql + traps
        {"naive_sql": "SELECT 1 AS n"},  # naive_sql only on dirty cases
    ],
)
def test_bad_cases_rejected(patch: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate({**GOOD, **patch})


def test_underspecified_needs_two_interpretations() -> None:
    interp = {
        "assumption": "a",
        "sql": "SELECT 1 AS n",
        "expected": {"n": {"match": "numeric", "value": 1}},
    }
    base = {
        "id": "under_99",
        "category": "underspecified",
        "question": "How are we doing?",
        "expected_behavior": "clarify_or_assume",
    }
    with pytest.raises(ValidationError):
        EvalCase.model_validate({**base, "interpretations": [interp]})
    EvalCase.model_validate({**base, "interpretations": [interp, interp]})


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalSet.model_validate({"cases": [GOOD, GOOD]})
