import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import judge

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import (
    AgentLoopDetectionMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
)
from deepeval.tracing import observe

from agent_instrumented import support_agent as _support_agent


@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)


plan_adherence = PlanAdherenceMetric(threshold=0.5, model=judge())
plan_quality = PlanQualityMetric(threshold=0.5, model=judge())
loop_detection = AgentLoopDetectionMetric(threshold=0.5)

dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?"),
])

for golden in dataset.evals_iterator(metrics=[plan_adherence, plan_quality, loop_detection]):
    support_agent(golden.input)
