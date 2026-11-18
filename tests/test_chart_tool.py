from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent.tools.chart_tool import MAX_BARS, ChartSpec, render_chart

PNG = b"\x89PNG\r\n\x1a\n"
COLS = ["region", "revenue", "orders"]
ROWS: list[list[Any]] = [
    ["North America", 2711157.26, 120],
    ["Europe", 1847427.0, 90],
    ["APAC", 1543061.0, 80],
]


@pytest.mark.parametrize("kind", ["bar", "line", "scatter"])
def test_renders_png(kind: str) -> None:
    r = render_chart(ChartSpec(kind=kind, x="region", y=["revenue"], title="Revenue"), COLS, ROWS)  # type: ignore[arg-type]
    assert r.ok, r.error
    assert r.png is not None and r.png.startswith(PNG)


def test_alt_text_names_extremes() -> None:
    r = render_chart(
        ChartSpec(kind="bar", x="region", y=["revenue"], title="Revenue 2025"), COLS, ROWS
    )
    assert "highest 2,711,157.26 at North America" in r.alt_text
    assert "lowest 1,543,061.00 at APAC" in r.alt_text


def test_multi_series_and_horizontal() -> None:
    spec = ChartSpec(kind="bar", x="region", y=["revenue", "orders"], title="t", horizontal=True)
    assert render_chart(spec, COLS, ROWS).ok


def test_unknown_column_rejected() -> None:
    r = render_chart(ChartSpec(kind="bar", x="region", y=["profit"], title="t"), COLS, ROWS)
    assert not r.ok and r.error is not None and "profit" in r.error


def test_non_numeric_series_rejected() -> None:
    r = render_chart(ChartSpec(kind="bar", x="revenue", y=["region"], title="t"), COLS, ROWS)
    assert not r.ok and r.error is not None and "numeric" in r.error


def test_too_many_bars_rejected() -> None:
    rows = [[f"c{i}", i, i] for i in range(MAX_BARS + 1)]
    r = render_chart(ChartSpec(kind="bar", x="region", y=["revenue"], title="t"), COLS, rows)
    assert not r.ok and r.error is not None and "too many" in r.error


def test_empty_result_rejected() -> None:
    assert not render_chart(
        ChartSpec(kind="line", x="region", y=["revenue"], title="t"), COLS, []
    ).ok


def test_mathtext_in_data_is_inert() -> None:
    # "$...$" would be parsed as TeX by matplotlib; untrusted labels must render literally
    rows: list[list[Any]] = [[r"$\frac{a}{b}$ \sqrt{2}", 1.0, 1], ["$$$", 2.0, 2]]
    assert render_chart(
        ChartSpec(kind="bar", x="region", y=["revenue"], title="$x$"), COLS, rows
    ).ok


def test_spec_is_strict() -> None:
    with pytest.raises(ValidationError):
        ChartSpec.model_validate({"kind": "pie", "x": "a", "y": ["b"], "title": "t"})
    with pytest.raises(ValidationError):
        ChartSpec.model_validate({"kind": "bar", "x": "a", "y": ["b"], "title": "t", "code": "x"})
