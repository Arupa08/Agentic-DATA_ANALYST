"""Sandbox escape attempts. Each must fail inside the child process, and nothing may
leak: no env vars, no database bytes, no files written outside the sandbox."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent.safety.sandbox import run_python

pytestmark = pytest.mark.usefixtures("warm_sandbox")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# reach the os module without importing it: the classic subclass walk
OS_GLOBALS = (
    "g = [c for c in ().__class__.__base__.__subclasses__() "
    "if c.__name__ == '_wrap_close'][0].__init__.__globals__\n"
)


def assert_blocked(code: str, *, marker: str = "") -> None:
    r = run_python(code, timeout_s=10)
    assert not r.ok, f"escape succeeded: stdout={r.stdout!r}"
    assert r.error is not None
    if marker:
        assert marker in r.error, r.error


@pytest.mark.parametrize(
    "module",
    [
        "os",
        "subprocess",
        "socket",
        "ctypes",
        "sqlite3",
        "shutil",
        "pathlib",
        "agent",
        "importlib",
        "sys",
    ],
)
def test_import_allowlist(module: str) -> None:
    assert_blocked(f"import {module}", marker="not allowed")


@pytest.mark.parametrize(
    "call", ["exec('1')", "eval('1')", "compile('1', 'x', 'exec')", "__import__('os')"]
)
def test_dangerous_builtins_removed(call: str) -> None:
    assert_blocked(call)


def test_os_system_via_subclass_walk() -> None:
    assert_blocked(OS_GLOBALS + "g['system']('echo pwned')", marker="blocked by sandbox")


def test_subprocess_via_real_import() -> None:
    assert_blocked(
        OS_GLOBALS + "sp = g['__builtins__']['__import__']('subprocess')\nsp.run(['echo', 'x'])",
        marker="blocked by sandbox",
    )


def test_network_via_real_socket() -> None:
    assert_blocked(
        OS_GLOBALS
        + "s = g['__builtins__']['__import__']('socket')\n"
        + "s.create_connection(('1.1.1.1', 80), timeout=2)",
        marker="blocked by sandbox",
    )


def test_network_via_pandas() -> None:
    assert_blocked(
        "import pandas as pd\npd.read_csv('https://example.com/x.csv')", marker="blocked by sandbox"
    )


def test_cannot_read_database_file(shop_db_path: Path) -> None:
    assert_blocked(f"open({str(shop_db_path)!r}, 'rb').read(16)", marker="blocked by sandbox")


def test_cannot_read_project_files_via_pandas() -> None:
    target = PROJECT_ROOT / ".env.example"
    assert_blocked(
        f"import pandas as pd\npd.read_csv({str(target)!r})", marker="blocked by sandbox"
    )


def test_cannot_list_project_directory() -> None:
    assert_blocked(
        OS_GLOBALS + f"print(g['listdir']({str(PROJECT_ROOT)!r}))", marker="blocked by sandbox"
    )


def test_cannot_write_outside_sandbox(tmp_path: Path) -> None:
    target = tmp_path / "pwned.txt"
    assert_blocked(f"open({str(target)!r}, 'w').write('x')", marker="blocked by sandbox")
    assert not target.exists()


def test_cannot_delete_outside_sandbox(tmp_path: Path) -> None:
    victim = tmp_path / "keep.txt"
    victim.write_text("keep")
    assert_blocked(OS_GLOBALS + f"g['remove']({str(victim)!r})", marker="blocked by sandbox")
    assert victim.exists()


def test_environment_has_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///secret/path.db")
    r = run_python(OS_GLOBALS + "print(dict(g['environ']))")
    assert r.ok  # reading the (clean) environment is allowed; it just holds nothing
    assert "sk-must-not-leak" not in r.stdout
    assert "OPENAI" not in r.stdout and "DATABASE_URL" not in r.stdout


def test_workdir_deleted_after_run() -> None:
    r = run_python(OS_GLOBALS + "print(g['getcwd']())")
    assert r.ok
    assert not Path(r.stdout.strip()).exists()


def test_infinite_loop_times_out() -> None:
    r = run_python("while True:\n    pass", timeout_s=3)
    assert not r.ok and r.timed_out
    assert r.elapsed_ms < 8_000


def test_output_flood_is_capped() -> None:
    r = run_python("print('x' * 10_000_000)")
    assert r.ok
    assert len(r.stdout) <= 20_000


def test_adding_a_permissive_hook_does_not_disable_ours() -> None:
    # audit hooks cannot be removed, so whatever user code adds, ours still runs
    assert_blocked(
        OS_GLOBALS
        + "try:\n    g['sys'].addaudithook(lambda *a: None)\nexcept Exception:\n    pass\n"
        + "g['system']('echo pwned')",
        marker="blocked by sandbox",
    )


@pytest.mark.skipif(sys.platform == "win32", reason="rlimits are POSIX-only; hosted runs on Linux")
def test_memory_bomb_is_stopped() -> None:
    r = run_python("x = bytearray(4 * 1024 ** 3)", memory_mb=512)
    assert not r.ok


@pytest.mark.skipif(sys.platform == "win32", reason="rlimits are POSIX-only; hosted runs on Linux")
def test_posix_limits_reported() -> None:
    r = run_python("result = 1")
    assert {"memory", "cpu", "file_size"} <= set(r.limits_enforced)
