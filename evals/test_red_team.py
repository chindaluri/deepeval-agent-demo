"""
Red-team the guarded ShopEasy support agent.

DeepTeam generates attacks for bias, toxicity, PII leakage, and prompt
leakage, then scores whether the guarded agent complied. The callback
goes through support_agent, so PromptInjectionGuard and TopicalGuard
run before Gemini does.

This script spends model calls. Run it on its own:

    python evals/test_red_team.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import judge

from deepteam import red_team
from deepteam.attacks.single_turn import PromptInjection, Roleplay
from deepteam.vulnerabilities import Bias, PIILeakage, PromptLeakage, Toxicity

from agent_instrumented import support_agent

PURPOSE = (
    "ShopEasy customer-support agent. It answers order status, refund, "
    "and shipping questions with tools, and refuses every other request."
)


async def model_callback(input: str, turns=None) -> str:
    return support_agent(input)


if __name__ == "__main__":
    red_team(
        model_callback=model_callback,
        vulnerabilities=[
            Bias(purpose=PURPOSE),
            Toxicity(purpose=PURPOSE),
            PIILeakage(purpose=PURPOSE),
            PromptLeakage(purpose=PURPOSE),
        ],
        attacks=[PromptInjection(), Roleplay()],
        simulator_model=judge(),
        evaluation_model=judge(),
        attacks_per_vulnerability_type=1,
        max_concurrent=1,
        target_purpose=PURPOSE,
    )
