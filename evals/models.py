"""Strict schema for eval cases. A malformed case is rejected at load time, never guessed at."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal[
    "simple_lookup", "multi_step", "dirty_data", "underspecified", "no_answer", "adversarial"
]
Behavior = Literal["answer", "clarify_or_assume", "no_data", "blocked"]

# what the plan says each category must do to pass
CATEGORY_BEHAVIOR: dict[str, str] = {
    "simple_lookup": "answer",
    "multi_step": "answer",
    "dirty_data": "answer",
    "underspecified": "clarify_or_assume",
    "no_answer": "no_data",
    "adversarial": "blocked",
}
# Phase 2 plan: 30 cases split across six categories
CATEGORY_COUNTS: dict[str, int] = {
    "simple_lookup": 5,
    "multi_step": 7,
    "dirty_data": 6,
    "underspecified": 4,
    "no_answer": 3,
    "adversarial": 5,
}
DATA_CARD_FLAWS = {"D1", "D2", "D3", "D4", "D5", "D6", "D7"}

DEFAULT_REL_TOL = 0.01  # numeric ±1%


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FieldSpec(_Strict):
    """One expected value and how a candidate answer is compared against it."""

    match: Literal["exact", "numeric", "set", "ordered"]
    value: str | float | list[str]
    rel_tol: float | None = Field(default=None, ge=0, le=0.5)
    abs_tol: float | None = Field(default=None, ge=0)
    aliases: list[str] = []

    @model_validator(mode="after")
    def _consistent(self) -> FieldSpec:
        if self.match == "numeric":
            if isinstance(self.value, bool) or not isinstance(self.value, int | float):
                raise ValueError("numeric field needs a number value")
        elif self.rel_tol is not None or self.abs_tol is not None:
            raise ValueError("tolerances apply only to numeric fields")
        if self.match == "exact" and not isinstance(self.value, str):
            raise ValueError("exact field needs a string value")
        if self.match in ("set", "ordered") and not isinstance(self.value, list):
            raise ValueError(f"{self.match} field needs a list value")
        if self.aliases and self.match != "exact":
            raise ValueError("aliases apply only to exact fields")
        return self

    @property
    def tolerance(self) -> tuple[float, float]:
        """(rel_tol, abs_tol); with neither given, the default is ±1% relative."""
        if self.rel_tol is None and self.abs_tol is None:
            return DEFAULT_REL_TOL, 0.0
        return self.rel_tol or 0.0, self.abs_tol or 0.0


def _check_expected(expected: dict[str, FieldSpec]) -> None:
    if not expected:
        raise ValueError("expected must name at least one field")
    multi_row = {f.match in ("set", "ordered") for f in expected.values()}
    if len(multi_row) > 1:
        raise ValueError("set/ordered fields cannot be mixed with scalar fields in one case")


class Interpretation(_Strict):
    """One reasonable reading of an underspecified question, with its own ground truth."""

    assumption: str
    sql: str
    expected: dict[str, FieldSpec]

    @model_validator(mode="after")
    def _valid(self) -> Interpretation:
        _check_expected(self.expected)
        return self


class EvalCase(_Strict):
    id: str = Field(pattern=r"^[a-z]+_\d{2}$")
    category: Category
    question: str = Field(min_length=5, max_length=500)
    expected_behavior: Behavior
    sql: str | None = None
    expected: dict[str, FieldSpec] = {}
    naive_sql: str | None = None
    interpretations: list[Interpretation] = []
    traps: list[str] = []
    rubric: list[str] = []
    must_not_contain: list[str] = []
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> EvalCase:
        if CATEGORY_BEHAVIOR[self.category] != self.expected_behavior:
            raise ValueError(
                f"{self.category} cases must expect {CATEGORY_BEHAVIOR[self.category]!r}"
            )
        if unknown := set(self.traps) - DATA_CARD_FLAWS:
            raise ValueError(f"unknown data-card flaws: {sorted(unknown)}")
        has_truth = self.sql is not None or bool(self.expected)
        if self.expected_behavior == "answer":
            if self.sql is None:
                raise ValueError("answer cases need ground-truth sql")
            _check_expected(self.expected)
            if self.interpretations:
                raise ValueError("answer cases take expected, not interpretations")
        elif has_truth:
            raise ValueError(f"{self.expected_behavior} cases carry no top-level sql/expected")
        if self.expected_behavior == "clarify_or_assume" and len(self.interpretations) < 2:
            raise ValueError("underspecified cases need at least two interpretations")
        if self.expected_behavior in ("no_data", "blocked") and not self.rubric:
            raise ValueError(f"{self.expected_behavior} cases need a rubric")
        if self.category == "dirty_data" and (self.naive_sql is None or not self.traps):
            raise ValueError("dirty_data cases need naive_sql and the flaws they trap")
        if self.naive_sql is not None and self.category != "dirty_data":
            raise ValueError("naive_sql is only for dirty_data cases")
        return self


class EvalSet(_Strict):
    cases: list[EvalCase]

    @model_validator(mode="after")
    def _unique_ids(self) -> EvalSet:
        ids = [c.id for c in self.cases]
        if dupes := {i for i in ids if ids.count(i) > 1}:
            raise ValueError(f"duplicate case ids: {sorted(dupes)}")
        return self


def load_cases(path: Path) -> EvalSet:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EvalSet.model_validate(raw)
