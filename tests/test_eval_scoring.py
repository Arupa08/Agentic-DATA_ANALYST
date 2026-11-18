from __future__ import annotations

from typing import Any

import pytest

from evals.models import FieldSpec
from evals.scoring import all_pass, match_field, score_fields


def spec(**kw: Any) -> FieldSpec:
    return FieldSpec.model_validate(kw)


@pytest.mark.parametrize(
    ("candidate", "ok"),
    [
        (1000.0, True),
        (1009.9, True),  # inside default ±1%
        (990.1, True),
        (1011, False),
        ("$1,005.00", True),  # formatted strings are parsed
        ("1,020", False),
        (None, False),
        (True, False),  # bools are not numbers
        ("about a thousand", False),
        (float("nan"), False),
    ],
)
def test_numeric_default_one_percent(candidate: Any, ok: bool) -> None:
    assert match_field(spec(match="numeric", value=1000), candidate) is ok


def test_numeric_abs_tol_for_percentages() -> None:
    s = spec(match="numeric", value=8.9, abs_tol=1.0)
    assert match_field(s, "9.5%")
    assert match_field(s, 7.95)
    assert not match_field(s, 10.0)


def test_exact_is_case_and_space_insensitive_with_aliases() -> None:
    s = spec(match="exact", value="Middle East & Africa", aliases=["MEA"])
    assert match_field(s, "  middle east  & africa ")
    assert match_field(s, "mea")
    assert not match_field(s, "Europe")


def test_set_ignores_order_but_not_duplicates_or_extras() -> None:
    s = spec(match="set", value=["a@x.com", "b@x.com"])
    assert match_field(s, ["B@x.com", "a@x.com"])
    assert not match_field(s, ["a@x.com", "a@x.com", "b@x.com"])
    assert not match_field(s, ["a@x.com", "b@x.com", "c@x.com"])
    assert not match_field(s, "a@x.com")


def test_ordered_requires_order() -> None:
    s = spec(match="ordered", value=["Electronics", "Home Appliances"])
    assert match_field(s, ["electronics", "home appliances"])
    assert not match_field(s, ["Home Appliances", "Electronics"])


def test_every_field_must_pass() -> None:
    expected = {
        "region": spec(match="exact", value="Europe"),
        "delta_pp": spec(match="numeric", value=8.9, abs_tol=1.0),
    }
    assert all_pass(score_fields(expected, {"region": "Europe", "delta_pp": 9.1}))
    assert not all_pass(score_fields(expected, {"region": "Europe", "delta_pp": 12}))
    assert not all_pass(score_fields(expected, {"region": "Europe"}))
    assert not all_pass([])
