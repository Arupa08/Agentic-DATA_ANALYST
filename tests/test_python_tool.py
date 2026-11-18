from __future__ import annotations

import pytest

from agent.safety.sandbox import TableInput, run_python

pytestmark = pytest.mark.usefixtures("warm_sandbox")

SALES = TableInput(columns=["region", "revenue"], rows=[["Europe", 10.0], ["APAC", 5.5]])


def test_prints_and_returns_dataframe() -> None:
    r = run_python("print(df.revenue.sum())\nresult = df.sort_values('revenue')", {"df": SALES})
    assert r.ok, r.error
    assert r.stdout.strip() == "15.5"
    assert r.result == {
        "columns": ["region", "revenue"],
        "rows": [["APAC", 5.5], ["Europe", 10.0]],
        "row_count": 2,
    }


def test_scalar_result() -> None:
    r = run_python("import numpy as np\nresult = float(np.mean(df.revenue))", {"df": SALES})
    assert r.ok and r.result == pytest.approx(7.75)


def test_open_figures_come_back_as_png() -> None:
    code = (
        "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.bar(df.region, df.revenue)"
    )
    r = run_python(code, {"df": SALES})
    assert r.ok, r.error
    assert len(r.figures) == 1 and r.figures[0][:8] == b"\x89PNG\r\n\x1a\n"


def test_user_error_reports_line() -> None:
    r = run_python("x = 1\ny = df['nope']", {"df": SALES})
    assert not r.ok
    assert r.error is not None and "KeyError" in r.error and "line 2" in r.error


def test_syntax_error() -> None:
    r = run_python("def (:")
    assert not r.ok and r.error is not None and "SyntaxError" in r.error


def test_rejects_bad_input_name() -> None:
    r = run_python("print(1)", {"__builtins__": SALES})
    assert not r.ok and r.error is not None and "identifier" in r.error
