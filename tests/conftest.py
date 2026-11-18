from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent.safety.sandbox import warm_up
from data.generate import Config, build_database


@pytest.fixture(scope="session")
def shop_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Full-scale DB built once per test session."""
    path: Path = tmp_path_factory.mktemp("data") / "shop.db"
    build_database(path, Config())
    return path


@pytest.fixture(scope="session")
def shop_db(shop_db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(f"file:{shop_db_path.as_posix()}?mode=ro", uri=True)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def warm_sandbox() -> None:
    """Pay matplotlib's one-time font-cache build outside the per-test 10s budget."""
    assert warm_up().ok
