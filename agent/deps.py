"""Everything the graph's nodes need, built once and injected (tests pass fakes)."""

from __future__ import annotations

from dataclasses import dataclass

from agent.config import Settings
from agent.llm import LangChainLLM, StructuredLLM
from agent.tools.sql_tool import SqlTool
from agent.tools.web_search import WebSearchTool, build_web_search


@dataclass(frozen=True)
class AgentDeps:
    settings: Settings
    llm: StructuredLLM
    sql: SqlTool
    web: WebSearchTool

    @property
    def schema_summary(self) -> str:
        return self.sql.schema_summary()  # cached per DB file


def build_deps(settings: Settings) -> AgentDeps:
    return AgentDeps(
        settings=settings,
        llm=LangChainLLM(settings),
        sql=SqlTool(settings.db_path),
        web=build_web_search(settings),
    )
