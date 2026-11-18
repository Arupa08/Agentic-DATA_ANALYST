"""Bootstrap that runs INSIDE the sandbox child process. Never import this module.

    python -I -X utf8 _sandbox_child.py <workdir> <memory_mb> <cpu_s>

Order matters:
  1. lower resource limits (POSIX) so nothing after this can exceed them
  2. load inputs and pre-import the allowed libraries while the process is still trusted
  3. install the audit hook (cannot be removed) that blocks network, processes, native
     code, the database and file access outside allowed roots
  4. run the user's code from user_code.py with stripped builtins and an import allowlist
  5. write result.json for the parent

Stdlib + pandas/numpy/matplotlib only; the agent package is deliberately not importable
from user code.
"""

import builtins
import io
import json
import os
import runpy
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, NoReturn

WORKDIR = os.path.realpath(sys.argv[1])
MEMORY_MB = int(sys.argv[2])
CPU_S = int(sys.argv[3])
USER_FILE = os.path.join(WORKDIR, "user_code.py")
MAX_STREAM_CHARS = 20_000
MAX_RESULT_ROWS = 5_000
MAX_FIGURES = 5

ALLOWED_IMPORTS = frozenset(
    {
        "pandas", "numpy", "matplotlib", "math", "statistics", "datetime", "json",
        "collections", "itertools", "functools", "re", "decimal", "fractions", "random",
        "typing", "dataclasses", "string", "textwrap", "calendar", "operator", "bisect",
        "heapq",
    }
)  # fmt: skip
BLOCKED_BUILTINS = ("exec", "eval", "compile", "breakpoint", "input", "help", "exit", "quit")

# ------------------------------------------------------------------ 1. resource limits
LIMITS = ["wall_clock"]
if sys.platform != "win32":
    import resource

    for res, value in (
        (resource.RLIMIT_AS, MEMORY_MB * 1024 * 1024),
        (resource.RLIMIT_CPU, CPU_S),
        (resource.RLIMIT_FSIZE, 50 * 1024 * 1024),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(res, (value, value))
    LIMITS += ["memory", "cpu", "file_size"]

# ------------------------------------------------------------------ 2. inputs + pre-imports
os.chdir(WORKDIR)
import numpy  # noqa: E402
import pandas  # noqa: E402

with open(os.path.join(WORKDIR, "inputs.json"), encoding="utf-8") as f:
    _inputs = json.load(f)
user_globals: dict[str, Any] = {
    name: pandas.DataFrame(t["rows"], columns=t["columns"]) for name, t in _inputs.items()
}

with open(USER_FILE, encoding="utf-8") as f:
    _uses_mpl = "matplotlib" in f.read()
if _uses_mpl:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot  # builds the font cache before the hook

# ------------------------------------------------------------------ 3. audit hook
_READ_ROOTS = [WORKDIR, sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]
if os.environ.get("MPLCONFIGDIR"):
    _READ_ROOTS.append(os.environ["MPLCONFIGDIR"])
if os.name == "posix":
    _READ_ROOTS.append("/usr/share/zoneinfo")
_READ_ROOTS = [os.path.normcase(os.path.realpath(p)) for p in _READ_ROOTS]
_WRITE_ROOTS = [os.path.normcase(WORKDIR)]
if os.environ.get("MPLCONFIGDIR"):
    _WRITE_ROOTS.append(os.path.normcase(os.path.realpath(os.environ["MPLCONFIGDIR"])))

_BLOCKED_PREFIXES = (
    "socket.", "subprocess.", "_posixsubprocess.", "os.system", "os.exec", "os.spawn",
    "os.posix_spawn", "os.fork", "os.forkpty", "os.startfile", "os.kill", "os.killpg",
    "os.putenv", "os.unsetenv", "os.chmod", "os.chown", "os.chflags", "os.link",
    "os.symlink", "os.setuid", "os.setgid", "ctypes.", "_winapi.", "winreg.", "msvcrt.",
    "sqlite3.", "urllib.", "http.", "ftplib.", "smtplib.", "poplib.", "imaplib.",
    "nntplib.", "telnetlib.", "webbrowser.", "pty.", "resource.setrlimit", "sys.addaudithook",
)  # fmt: skip
_WRITE_EVENTS = {
    "os.remove", "os.rmdir", "os.rename", "os.mkdir", "os.utime", "os.truncate",
    "shutil.rmtree", "shutil.move", "shutil.copyfile", "shutil.copytree",
}  # fmt: skip
_LIST_EVENTS = {"os.listdir", "os.scandir", "glob.glob", "glob.glob/2"}
_DENY_IMPORT_EVENT = {
    "ctypes", "_ctypes", "socket", "_socket", "ssl", "_ssl", "sqlite3", "_sqlite3",
    "subprocess", "multiprocessing", "_winapi", "winreg",
}  # fmt: skip
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_in_hook = False


class SandboxViolation(PermissionError):
    pass


def _within(path: Any, roots: list[str]) -> bool:
    try:
        p = os.path.normcase(os.path.realpath(os.fspath(path)))
    except (TypeError, ValueError):
        return False
    for root in roots:
        try:
            if os.path.commonpath([p, root]) == root:
                return True
        except ValueError:  # different drives on Windows
            continue
    return False


def _is_write(mode: object, flags: object) -> bool:
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _audit(event: str, args: tuple[Any, ...]) -> None:
    global _in_hook
    if _in_hook:
        return
    _in_hook = True
    try:
        if event.startswith(_BLOCKED_PREFIXES):
            raise SandboxViolation(f"blocked by sandbox: {event}")
        if event == "import" and args and str(args[0]).split(".")[0] in _DENY_IMPORT_EVENT:
            raise SandboxViolation(f"blocked by sandbox: import {args[0]}")
        if event == "open":
            path, mode, flags = [*args, None, None, None][:3]
            if isinstance(path, int):
                return
            roots = _WRITE_ROOTS if _is_write(mode, flags) else _READ_ROOTS
            if not _within(path, roots):
                raise SandboxViolation("blocked by sandbox: file access outside the sandbox")
        elif event in _WRITE_EVENTS:
            for path in args[:2]:
                if isinstance(path, str | bytes | os.PathLike) and not _within(path, _WRITE_ROOTS):
                    raise SandboxViolation(f"blocked by sandbox: {event} outside the sandbox")
        elif event in _LIST_EVENTS and args and isinstance(args[0], str | bytes | os.PathLike):
            if not _within(args[0], _READ_ROOTS):
                raise SandboxViolation(f"blocked by sandbox: {event} outside the sandbox")
    finally:
        _in_hook = False


# ------------------------------------------------------------------ 4. user code
class _CappedStream(io.StringIO):
    def write(self, s: str) -> int:
        room = MAX_STREAM_CHARS - self.tell()
        if room > 0:
            super().write(s[:room])
        return len(s)


def _guarded_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> ModuleType:
    if level != 0 or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in the sandbox")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _blocked_builtin(name: str) -> Callable[..., NoReturn]:
    def _blocked(*_args: object, **_kwargs: object) -> NoReturn:
        raise SandboxViolation(f"{name}() is not available in the sandbox")

    return _blocked


safe_builtins: dict[str, Any] = dict(vars(builtins))
safe_builtins["__import__"] = _guarded_import
for _name in BLOCKED_BUILTINS:
    safe_builtins[_name] = _blocked_builtin(_name)
user_globals["__builtins__"] = safe_builtins

out, err = _CappedStream(), _CappedStream()
real_stdout, real_stderr = sys.stdout, sys.stderr
outcome: dict[str, Any] = {"ok": True, "error": None, "result": None, "limits": LIMITS}

sys.addaudithook(_audit)
sys.stdout, sys.stderr = out, err
try:
    namespace = runpy.run_path(USER_FILE, init_globals=user_globals, run_name="__sandbox__")
except BaseException as exc:  # user code may raise anything, including SystemExit
    outcome["ok"] = False
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == USER_FILE]
    where = f" (line {frames[-1].lineno})" if frames else ""
    outcome["error"] = f"{type(exc).__name__}: {exc}{where}"
    namespace = {}
finally:
    sys.stdout, sys.stderr = real_stdout, real_stderr


# ------------------------------------------------------------------ 5. result.json
def _jsonable(value: Any) -> Any:
    if isinstance(value, pandas.Series):
        value = value.to_frame()
    if isinstance(value, pandas.DataFrame):
        frame = value.reset_index() if not isinstance(value.index, pandas.RangeIndex) else value
        head = frame.head(MAX_RESULT_ROWS)
        return {
            "columns": [str(c) for c in head.columns],
            "rows": json.loads(head.to_json(orient="values", date_format="iso")),
            "row_count": len(frame),
        }
    if isinstance(value, numpy.generic):
        return value.item()
    return json.loads(json.dumps(value, default=str))


if outcome["ok"] and "result" in namespace:
    try:
        outcome["result"] = _jsonable(namespace["result"])
    except Exception as exc:
        outcome["ok"] = False
        outcome["error"] = f"could not serialize `result`: {type(exc).__name__}: {exc}"

if "matplotlib.pyplot" in sys.modules:
    plt = sys.modules["matplotlib.pyplot"]
    for i, num in enumerate(plt.get_fignums()[:MAX_FIGURES]):
        plt.figure(num).savefig(os.path.join(WORKDIR, f"figure_{i + 1}.png"), dpi=100)

outcome["stdout"] = out.getvalue()
outcome["stderr"] = err.getvalue()
with open(os.path.join(WORKDIR, "result.json"), "w", encoding="utf-8") as f:
    json.dump(outcome, f, default=str)
