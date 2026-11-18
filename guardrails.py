"""
guardrails.py
=============
Runtime safety checks for the ShopEasy agents.

DeepEval 4.x moved red teaming into DeepTeam. These guards run in the
request path: an unsafe input never reaches the model, and an unsafe
output is replaced before the customer sees it.

Gemini is the guard judge, matching the eval suite.
"""

from llm import judge

from deepteam.guardrails import Guardrails
from deepteam.guardrails.guards import (
    IllegalGuard,
    PrivacyGuard,
    PromptInjectionGuard,
    TopicalGuard,
    ToxicityGuard,
)

# One topic string keeps refund-policy questions on-topic. TopicalGuard is
# an input guard only: its output prompt treats the word "policy" as unsafe,
# which would block every legitimate refund answer.
ALLOWED_TOPICS = [
    "ShopEasy customer support: order status, delivery, refunds, returns, and shipping",
]

REFUSAL = "I can only help with ShopEasy orders, refunds, and shipping."

guardrails = Guardrails(
    input_guards=[
        PromptInjectionGuard(),
        TopicalGuard(allowed_topics=ALLOWED_TOPICS),
    ],
    output_guards=[
        ToxicityGuard(),
        PrivacyGuard(),
        IllegalGuard(),
    ],
    evaluation_model=judge(),
    sample_rate=1.0,
)


def _as_text(content) -> str:
    """LangChain sometimes returns a list of content blocks instead of a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                text = getattr(block, "text", None)
                parts.append(text if isinstance(text, str) else str(block))
        return "".join(parts)
    return str(content)


def _is_unsafe(result) -> bool:
    """Block only an explicit unsafe verdict.

    DeepTeam also marks borderline and uncertain as breached, which refused
    ordinary order questions such as "Where is my order ORD-1042?".
    """
    for verdict in result.verdicts:
        level = getattr(verdict, "safety_level", None)
        value = getattr(level, "value", level)
        if str(value).lower() == "unsafe":
            return True
    return False


def screen_input(user_input: str) -> str | None:
    """Return a refusal when the input is unsafe, otherwise None."""
    if _is_unsafe(guardrails.guard_input(user_input)):
        return REFUSAL
    return None


def screen_output(user_input: str, reply) -> str:
    """Return the reply, or a refusal when the reply is unsafe."""
    text = _as_text(reply)
    if _is_unsafe(guardrails.guard_output(input=user_input, output=text)):
        return REFUSAL
    return text
