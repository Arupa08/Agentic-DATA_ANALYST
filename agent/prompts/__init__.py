"""Versioned prompt templates. Templates use $placeholders (string.Template) so braces
in schema text or data can never be mistaken for template fields."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).resolve().parent
PROMPT_VERSION = "v1"


@cache
def _template(name: str, version: str) -> Template:
    return Template((PROMPT_DIR / f"{name}_{version}.md").read_text(encoding="utf-8"))


def render(name: str, version: str = PROMPT_VERSION, **values: str) -> str:
    """Fill a template; every placeholder must be supplied (KeyError otherwise)."""
    return _template(name, version).substitute(values)


def raw(name: str, version: str = PROMPT_VERSION) -> str:
    return _template(name, version).template
