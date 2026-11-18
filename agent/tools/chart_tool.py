"""Chart tool: renders a PNG from a declarative spec plus a result table.

The model only chooses the spec (kind, columns, title); rendering is fixed code, so no
model-written code runs here. Colors and mark rules follow the dataviz skill's
validated default palette: fixed categorical order, one y-axis only, a legend only for
2+ series, direct labels on lines, recessive grid.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Literal

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

# validated default palette, light surface (dataviz references/palette.md)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

MAX_BARS = 30
MAX_BAR_SERIES = 4
MAX_POINTS = 2_000
LINE_PX = 2.0
DPI = 150


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["bar", "line", "scatter"]
    x: str
    y: list[str] = Field(min_length=1, max_length=len(SERIES))
    title: str = Field(min_length=1, max_length=120)
    x_label: str | None = Field(default=None, max_length=60)
    y_label: str | None = Field(default=None, max_length=60)
    horizontal: bool = False  # bar only: better for long category names


class ChartResult(BaseModel):
    ok: bool
    png: bytes | None = None
    alt_text: str = ""
    error: str | None = None


class ChartError(ValueError):
    pass


def _safe_text(value: Any) -> str:
    # data values are untrusted: disable matplotlib mathtext ($...$) and cap length
    return str(value).replace("$", r"\$")[:60]


def _frame(spec: ChartSpec, columns: list[str], rows: list[list[Any]]) -> pd.DataFrame:
    missing = [c for c in (spec.x, *spec.y) if c not in columns]
    if missing:
        raise ChartError(f"columns not in result: {missing}; available: {columns}")
    if not rows:
        raise ChartError("no rows to chart")
    df = pd.DataFrame(rows, columns=columns)[[spec.x, *spec.y]]
    for col in spec.y:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            raise ChartError(f"column {col!r} has no numeric values")
    if spec.kind == "bar":
        if len(df) > MAX_BARS:
            raise ChartError(f"{len(df)} bars is too many; aggregate or keep the top {MAX_BARS}")
        if len(spec.y) > MAX_BAR_SERIES:
            raise ChartError(f"at most {MAX_BAR_SERIES} series in a bar chart")
    elif len(df) > MAX_POINTS:
        raise ChartError(f"{len(df)} points is too many; aggregate to at most {MAX_POINTS}")
    if spec.kind == "line":
        df = df.sort_values(spec.x, kind="stable")
    return df


def _compact(value: float, _pos: int | None = None) -> str:
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:,.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _style(ax: Any, value_axis: Literal["x", "y"]) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(axis=value_axis, color=GRID, linewidth=0.6)
    (ax.xaxis if value_axis == "x" else ax.yaxis).set_major_formatter(FuncFormatter(_compact))
    ax.set_axisbelow(True)


def render_chart(spec: ChartSpec, columns: list[str], rows: list[list[Any]]) -> ChartResult:
    try:
        df = _frame(spec, columns, rows)
    except ChartError as exc:
        return ChartResult(ok=False, error=str(exc))

    fig = Figure(figsize=(8, 4.5), dpi=DPI, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    lw = LINE_PX * 72 / DPI
    x_labels = [_safe_text(v) for v in df[spec.x]]
    multi = len(spec.y) > 1

    if spec.kind == "bar":
        n = len(spec.y)
        width = 0.8 / n
        positions = range(len(df))
        for i, col in enumerate(spec.y):
            offsets = [p - 0.4 + width * (i + 0.5) for p in positions]
            bar = ax.barh if spec.horizontal else ax.bar
            bar(offsets, df[col], width, color=SERIES[i], label=_safe_text(col),
                edgecolor=SURFACE, linewidth=lw)  # fmt: skip  # 2px surface gap between fills
        if spec.horizontal:
            ax.set_yticks(list(positions), x_labels)
            ax.invert_yaxis()
            _style(ax, "x")
        else:
            ax.set_xticks(
                list(positions),
                x_labels,
                rotation=30 if len(df) > 8 else 0,
                ha="right" if len(df) > 8 else "center",
            )
            _style(ax, "y")
    else:
        numeric_x = pd.api.types.is_numeric_dtype(df[spec.x])
        x_vals: Any = df[spec.x] if numeric_x else x_labels
        last_x = x_vals.iloc[-1] if numeric_x else len(df) - 1
        for i, col in enumerate(spec.y):
            if spec.kind == "line":
                ax.plot(x_vals, df[col], color=SERIES[i], linewidth=lw * 1.5, label=_safe_text(col))
                if multi and len(spec.y) <= 4:  # direct label at the line end
                    ax.annotate(
                        _safe_text(col),
                        (last_x, df[col].iloc[-1]),
                        xytext=(4, 0),
                        textcoords="offset points",
                        color=INK_SECONDARY,
                        fontsize=8,
                        va="center",
                    )
            else:
                ax.scatter(x_vals, df[col], s=(8 * 72 / DPI) ** 2 * 4, color=SERIES[i],
                           edgecolors=SURFACE, linewidths=lw, label=_safe_text(col))  # fmt: skip
        if not numeric_x and len(df) > 12:
            step = max(1, len(df) // 12)
            ax.set_xticks(range(0, len(df), step), x_labels[::step], rotation=30, ha="right")
        _style(ax, "y")

    ax.set_title(_safe_text(spec.title), loc="left", color=INK_PRIMARY, fontsize=11, pad=12)
    if spec.x_label:
        (ax.set_ylabel if spec.horizontal else ax.set_xlabel)(
            _safe_text(spec.x_label), color=INK_SECONDARY, fontsize=9
        )
    if spec.y_label:
        (ax.set_xlabel if spec.horizontal else ax.set_ylabel)(
            _safe_text(spec.y_label), color=INK_SECONDARY, fontsize=9
        )
    if multi:
        ax.legend(
            frameon=False,
            fontsize=8,
            labelcolor=INK_SECONDARY,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    log.info("chart_rendered", extra={"kind": spec.kind, "points": len(df), "series": len(spec.y)})
    return ChartResult(ok=True, png=buf.getvalue(), alt_text=_alt_text(spec, df))


def _alt_text(spec: ChartSpec, df: pd.DataFrame) -> str:
    parts = [f"{spec.kind} chart: {spec.title}. {len(df)} {spec.x} values."]
    for col in spec.y:
        series = df[col].dropna()
        if series.empty:
            continue
        hi = df.loc[series.idxmax(), spec.x]
        lo = df.loc[series.idxmin(), spec.x]
        parts.append(
            f"{col}: highest {series.max():,.2f} at {hi}, lowest {series.min():,.2f} at {lo}."
        )
    return " ".join(parts)
