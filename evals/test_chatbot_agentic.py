import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import judge

from deepeval.evaluate import evaluate
from deepeval.metrics import GoalAccuracyMetric, ToolUseMetric
from deepeval.test_case import ConversationalTestCase, ToolCall, Turn

from chatbot import chat

USER_MESSAGES = [
    "Hi! I placed an order last week, the order ID is ORD-1042.",
    "Is it going to arrive on time?",
    "What was the ETA you just mentioned?",
    "Can I upgrade to express shipping?",
]

AVAILABLE_TOOLS = [
    ToolCall(
        name="get_order_status",
        description="Look up the shipping status of an order by its ID.",
    ),
    ToolCall(
        name="get_refund_policy",
        description="Return the refund policy for a product category.",
    ),
]


def _as_tool_calls(raw_calls):
    return [
        ToolCall(
            name=call["name"],
            input_parameters=call["args"],
            output=call["result"],
        )
        for call in raw_calls
    ]


turns = []
history = []
for user_msg in USER_MESSAGES:
    reply, history, tools_called = chat(user_msg, history)
    turns.append(Turn(role="user", content=user_msg))
    turns.append(Turn(
        role="assistant",
        content=reply,
        tools_called=_as_tool_calls(tools_called),
    ))

tool_use = ToolUseMetric(
    available_tools=AVAILABLE_TOOLS,
    threshold=0.5,
    model=judge(),
)
goal_accuracy = GoalAccuracyMetric(threshold=0.5, model=judge())

test_case = ConversationalTestCase(
    turns=turns,
    chatbot_role=(
        "Friendly ShopEasy customer-support agent. Help with orders, "
        "refunds, and shipping, and stay on that job."
    ),
    expected_outcome=(
        "The customer learns that ORD-1042 is shipped with an ETA, "
        "and whether the shipping method can be changed."
    ),
)

evaluate(test_cases=[test_case], metrics=[tool_use, goal_accuracy])
