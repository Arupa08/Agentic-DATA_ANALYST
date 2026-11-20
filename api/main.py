"""POST /ask: the agent over HTTP, behind the controls a public URL needs.

    uv run uvicorn api.main:app --port 8000

Every response carries security headers; browsers may only call it from the UI's origin;
each client IP is rate-limited; request bodies over a few KB are refused before they are
read; at most `API_MAX_CONCURRENT` questions run at once (others get 503, not a queue).
Users see short generic errors; details and stack traces go to the server log only.

The agent is built at startup from `agent.config.Settings`, so the server refuses to
start when a required model key is missing.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agent.graph import MAX_QUESTION_CHARS, AgentResponse
from agent.llm import RateLimited
from api.config import APISettings

log = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cross-Origin-Resource-Policy": "same-site",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}
# the interactive docs page loads its own scripts and styles, so it gets a looser policy
DOCS_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'none'"
)
HSTS = "max-age=31536000; includeSubDomains"


class AskAgent(Protocol):
    def ask(self, question: str, *, tags: tuple[str, ...] = ()) -> AgentResponse: ...


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


def _error(status: int, message: str, **headers: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers=headers or None)


class BodySizeLimit:
    """Refuses bodies over `max_bytes`: by Content-Length up front, and otherwise (a
    chunked request has none) by reading at most `max_bytes` before the app sees any of
    it. The limit is a few KB, so holding the body in memory costs nothing."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        too_large = _error(413, "request too large")
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            await too_large(scope, receive, send)
            return
        body = b""
        while True:
            message = await receive()
            if message["type"] != "http.request":  # client went away
                return
            body += message.get("body", b"")
            if len(body) > self.max_bytes:
                await too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def _build_agent() -> AskAgent:
    from agent.config import get_settings
    from agent.deps import build_deps
    from agent.graph import Agent
    from agent.logging_setup import configure_logging
    from agent.safety.sandbox import warm_up

    settings = get_settings()  # raises if a required key is missing: refuse to start
    configure_logging(settings.log_level)
    warm_up()
    return Agent(build_deps(settings))


def create_app(
    agent: AskAgent | None = None,
    settings: APISettings | None = None,
    agent_factory: Callable[[], AskAgent] = _build_agent,
) -> FastAPI:
    api = settings if settings is not None else APISettings()
    holder: dict[str, AskAgent] = {}
    if agent is not None:
        holder["agent"] = agent

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if "agent" not in holder:
            holder["agent"] = agent_factory()
        yield

    app = FastAPI(
        title="NL Analytics Agent",
        version="0.1.0",
        summary="Ask business questions about a synthetic shop's data in plain English.",
        lifespan=lifespan,
        docs_url="/docs" if api.docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if api.docs else None,
    )
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    slots = threading.BoundedSemaphore(api.max_concurrent)

    app.add_middleware(BodySizeLimit, max_bytes=api.max_body_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
        max_age=600,
    )

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path == "/docs":
            response.headers["Content-Security-Policy"] = DOCS_CSP
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = HSTS
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        # field and problem only: the submitted value is never echoed back
        details = [
            {"field": ".".join(str(p) for p in e["loc"][1:]) or "body", "problem": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse({"error": "invalid request", "details": details}, status_code=422)

    @app.exception_handler(RateLimitExceeded)
    async def too_many(_: Request, __: RateLimitExceeded) -> JSONResponse:
        return _error(429, "too many requests, try again in a minute", **{"Retry-After": "60"})

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        error_id = uuid.uuid4().hex[:12]
        log.error("api_error", extra={"error_id": error_id}, exc_info=exc)
        return JSONResponse({"error": "internal error", "error_id": error_id}, status_code=500)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/ask",
        response_model=AgentResponse,
        responses={
            413: {"description": "Request body too large"},
            422: {"description": "Invalid question"},
            429: {"description": "Rate limit for this client reached"},
            503: {"description": "Busy, or the model provider is rate-limited"},
        },
    )
    @limiter.limit(api.rate_limit)
    def ask(request: Request, body: AskRequest) -> Any:
        """Answer one question: the answer, assumptions and caveats, the SQL that ran,
        charts (base64 PNG), the plan, retries and critic verdicts, cost and latency,
        and the LangSmith trace id when tracing is on."""
        if not slots.acquire(blocking=False):
            busy = "busy answering other questions, try again shortly"
            return _error(503, busy, **{"Retry-After": "30"})
        try:
            response = holder["agent"].ask(body.question, tags=("api",))
        except ValueError:
            return _error(422, f"the question must be 1 to {MAX_QUESTION_CHARS} characters")
        except RateLimited:
            return _error(503, "the model is rate-limited or out of quota, try again later")
        finally:
            slots.release()
        return Response(response.model_dump_json(), media_type="application/json")

    return app


app = create_app()
