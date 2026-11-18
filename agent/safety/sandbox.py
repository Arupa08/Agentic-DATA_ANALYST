"""Run LLM-written Python in a separate, locked-down process. Never in this process.

Isolation, from the outside in:
  - separate interpreter (`python -I`): no PYTHON* env, no user site, no cwd on sys.path
  - clean environment: no API keys, no DATABASE_URL; only what numpy/matplotlib need
  - fresh temp working directory, deleted after every run
  - wall-clock timeout (all platforms); memory / CPU / file-size rlimits (POSIX only)
  - inside the child, see _sandbox_child.py: an audit hook blocking network, processes,
    native code, sqlite and file access outside the sandbox, plus an import allowlist

On Windows only the wall-clock limit is OS-enforced. The hosted deployment runs in a
Linux container (Phase 7), where all limits apply. Upgrade path: E2B or a per-run container.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess  # nosec B404 - required to launch the isolated child interpreter
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)

CHILD_SCRIPT = Path(__file__).with_name("_sandbox_child.py")
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MEMORY_MB = 1024
MAX_CODE_CHARS = 20_000
MAX_INPUT_ROWS = 50_000
MAX_RESULT_BYTES = 5 * 1024 * 1024
MAX_FIGURE_BYTES = 5 * 1024 * 1024
MAX_FIGURES = 5
MPL_CACHE = Path(tempfile.gettempdir()) / "nl-agent-sandbox-mpl"


class TableInput(BaseModel):
    columns: list[str]
    rows: list[list[Any]]


class SandboxResult(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    result: Any = None
    figures: list[bytes] = []
    timed_out: bool = False
    elapsed_ms: int = 0
    limits_enforced: list[str] = []


def _child_env(workdir: str) -> dict[str, str]:
    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(MPL_CACHE),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HOME": workdir,
        "TMPDIR": workdir,
        "TEMP": workdir,
        "TMP": workdir,
    }
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _validate_inputs(inputs: dict[str, TableInput]) -> None:
    for name, table in inputs.items():
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"input name {name!r} must be a public Python identifier")
        if len(table.rows) > MAX_INPUT_ROWS:
            raise ValueError(f"input {name!r} has more than {MAX_INPUT_ROWS} rows")


def run_python(
    code: str,
    inputs: dict[str, TableInput] | None = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> SandboxResult:
    """Run `code` with each input exposed as a pandas DataFrame variable.

    The code may set `result` (DataFrame, Series, scalar or JSON-able value) and may save
    PNGs or draw matplotlib figures; both come back in the SandboxResult.
    """
    if len(code) > MAX_CODE_CHARS:
        return SandboxResult(ok=False, error=f"code longer than {MAX_CODE_CHARS} characters")
    inputs = inputs or {}
    try:
        _validate_inputs(inputs)
    except ValueError as exc:
        return SandboxResult(ok=False, error=str(exc))

    MPL_CACHE.mkdir(exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="nl-agent-sandbox-")
    start = time.monotonic()
    try:
        Path(workdir, "user_code.py").write_text(code, encoding="utf-8")
        Path(workdir, "inputs.json").write_text(
            json.dumps({k: v.model_dump() for k, v in inputs.items()}, default=str),
            encoding="utf-8",
        )
        cmd = [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            str(CHILD_SCRIPT),
            workdir,
            str(memory_mb),
            str(int(timeout_s) + 1),
        ]
        try:
            proc = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, no shell
                cmd,
                cwd=workdir,
                env=_child_env(workdir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.info("sandbox_timeout", extra={"timeout_s": timeout_s})
            return SandboxResult(
                ok=False,
                error=f"code timed out after {timeout_s:g}s",
                timed_out=True,
                elapsed_ms=_ms(start),
            )
        return _collect(Path(workdir), proc, start)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _collect(
    workdir: Path, proc: subprocess.CompletedProcess[bytes], start: float
) -> SandboxResult:
    result_path = workdir / "result.json"
    if not result_path.exists() or result_path.stat().st_size > MAX_RESULT_BYTES:
        # crashed before writing a result (e.g. killed by a memory or CPU limit)
        log.warning(
            "sandbox_crash",
            extra={
                "exit_code": proc.returncode,
                "stderr_tail": proc.stderr[-2000:].decode(errors="replace"),
            },
        )
        return SandboxResult(
            ok=False,
            error=(
                f"sandbox process failed (exit code {proc.returncode}); "
                "it may have hit a resource limit"
            ),
            elapsed_ms=_ms(start),
        )
    # result.json is written in a directory the user code controls: parse it as untrusted
    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        outcome = raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        outcome = {}
    figures = [
        p.read_bytes()
        for p in sorted(workdir.glob("*.png"))[:MAX_FIGURES]
        if p.is_file() and p.stat().st_size <= MAX_FIGURE_BYTES
    ]
    ok = bool(outcome.get("ok")) and proc.returncode == 0
    log.info("sandbox_done", extra={"ok": ok, "figures": len(figures), "elapsed_ms": _ms(start)})
    return SandboxResult(
        ok=ok,
        stdout=str(outcome.get("stdout", ""))[:20_000],
        stderr=str(outcome.get("stderr", ""))[:20_000],
        error=None if ok else str(outcome.get("error") or f"exit code {proc.returncode}")[:2_000],
        result=outcome.get("result"),
        figures=figures,
        elapsed_ms=_ms(start),
        limits_enforced=[str(x) for x in outcome.get("limits", [])][:10],
    )


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def warm_up() -> SandboxResult:
    """Pay the one-time costs (bytecode, matplotlib font cache) outside any request's
    10s budget. Call once at startup."""
    return run_python("import matplotlib.pyplot\nimport pandas", timeout_s=120)
