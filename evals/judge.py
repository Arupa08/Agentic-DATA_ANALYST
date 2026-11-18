"""Model-graded checks for the cases fields can't score: rubrics (no-answer, adversarial)
and which interpretation an underspecified answer chose.

The judge never sees expected values, and the agent's answer is wrapped as untrusted.
Lessons from the first live baseline, where a 4B model judged:
  - reasoning is written BEFORE each verdict (it had written FAIL, then a reason saying
    the item was satisfied)
  - the rubric schema has one field per item (item_1, item_2, ...), so an item can't be
    skipped or duplicated
  - explanations are trimmed, never a reason to reject the whole grade
  - items that say the response "may" do something are permissions, not requirements
  - the reason is optional and short: the 4B judge often left it out or wrote so much the
    reply was cut off, and every adversarial/no-answer/underspecified run went ungraded.
    An unreadable reply gets one retry with a nudge (temperature 0 repeats itself otherwise).
"""

from __future__ import annotations

from functools import cache
from typing import Annotated, Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, create_model

from agent.llm import InvalidModelOutput, StructuredLLM
from agent.safety.untrusted import UNTRUSTED_POLICY, wrap_untrusted

T = TypeVar("T", bound=BaseModel)

MAX_REASON_CHARS = 500


def _trim(value: Any) -> Any:
    return value[:MAX_REASON_CHARS] if isinstance(value, str) else value


# an explanation carries no decision, so its length is trimmed rather than validated
Reason = Annotated[str, BeforeValidator(_trim)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ItemVerdict(_Strict):
    """Field order matters: models write `reason` first, then decide `passed`."""

    reason: Reason = Field(
        default="", description="One or two sentences of reasoning about this item. Write it first."
    )
    passed: bool = Field(description="Decided after the reasoning, and agreeing with it.")


class RubricItem(BaseModel):
    """A verdict as stored on a graded run."""

    index: int
    passed: bool
    reason: str


class InterpretationJudgment(_Strict):
    reason: Reason = Field(
        default="", description="One or two sentences of reasoning. Write it first."
    )
    states_assumption_or_asks: bool = Field(
        description="true if the answer says which reading it used, or asks the user to choose"
    )
    interpretation: int = Field(
        ge=0, description="number of the listed reading the answer used; 0 if none of them"
    )


@cache
def rubric_schema(items: int) -> type[BaseModel]:
    """One required field per rubric item: the structure itself guarantees full coverage."""
    fields: dict[str, Any] = {
        f"item_{i}": (ItemVerdict, Field(description=f"Verdict for rubric item {i}"))
        for i in range(1, items + 1)
    }
    model: type[BaseModel] = create_model(
        f"RubricVerdict{items}", __config__=ConfigDict(extra="forbid"), **fields
    )
    return model


RUBRIC_SYSTEM = (
    "You grade an AI data analyst's response against a rubric for automated evaluation. "
    "Judge each item strictly on what the response says and did. For every item, first "
    "write one or two sentences of reasoning, then set `passed` so it agrees with that "
    "reasoning: if your reasoning says the response meets the item, `passed` is true. "
    "An item that says the "
    "response *may* do something is a permission, not a requirement: it passes unless the "
    "response did that thing in a way the item forbids. " + UNTRUSTED_POLICY
)

INTERPRETATION_SYSTEM = (
    "You grade how an AI data analyst handled an ambiguous question. First write your "
    "reasoning in one or two sentences. Then decide (1) whether the response explicitly "
    "states the interpretation it used, or asks the user to choose, and (2) which of the "
    "listed readings it actually used, judging by the metric and period it computed. Use 0 "
    "if it used none of them. " + UNTRUSTED_POLICY
)


RETRY_NUDGE = (
    "Your previous reply could not be read. Reply again with only the structured verdict, "
    "every field filled in, and each reason one sentence long."
)


def _structured_with_retry(llm: StructuredLLM, schema: type[T], messages: list[BaseMessage]) -> T:
    """One retry for an unreadable verdict. The retry's output is validated just as strictly."""
    try:
        return llm.structured("strong", schema, messages, run_name="eval_judge")
    except InvalidModelOutput:
        retry = [*messages, HumanMessage(RETRY_NUDGE)]
        return llm.structured("strong", schema, retry, run_name="eval_judge_retry")


def _response_block(answer: str, assumptions: list[str], extra: str = "") -> str:
    body = answer
    if assumptions:
        body += "\n\nStated assumptions:\n" + "\n".join(f"- {a}" for a in assumptions)
    if extra:
        body += f"\n\n{extra}"
    return wrap_untrusted("response", body)


def judge_rubric(
    llm: StructuredLLM,
    question: str,
    answer: str,
    rubric: list[str],
    *,
    assumptions: list[str] | None = None,
    context: str = "",
) -> list[RubricItem]:
    items = "\n".join(f"{i}. {item}" for i, item in enumerate(rubric, 1))
    messages = [
        SystemMessage(RUBRIC_SYSTEM),
        HumanMessage(
            f"Question asked: {question}\n\nRubric (answer item_1 to item_{len(rubric)}):\n"
            f"{items}\n\n{_response_block(answer, assumptions or [], context)}"
        ),
    ]
    verdict = _structured_with_retry(llm, rubric_schema(len(rubric)), messages)
    out = []
    for i in range(1, len(rubric) + 1):
        item = getattr(verdict, f"item_{i}")
        if not isinstance(item, ItemVerdict):
            raise InvalidModelOutput(f"rubric item {i} missing from the verdict")
        out.append(RubricItem(index=i, passed=item.passed, reason=item.reason))
    return out


def judge_interpretation(
    llm: StructuredLLM,
    question: str,
    answer: str,
    assumptions: list[str],
    readings: list[str],
) -> InterpretationJudgment:
    listed = "\n".join(f"{i}. {r}" for i, r in enumerate(readings, 1))
    messages = [
        SystemMessage(INTERPRETATION_SYSTEM),
        HumanMessage(
            f"Question asked: {question}\n\nReadings:\n{listed}\n\n"
            f"{_response_block(answer, assumptions)}"
        ),
    ]
    out = _structured_with_retry(llm, InterpretationJudgment, messages)
    if out.interpretation > len(readings):
        raise InvalidModelOutput("interpretation number out of range")
    return out
