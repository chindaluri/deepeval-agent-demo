# DeepEval ShopEasy Demo

A small customer-support agent, evaluated with [DeepEval](https://github.com/confident-ai/deepeval). The same Gemini model answers customers, judges the evals, embeds the RAG documents, and scores the guardrails.

The agent is ShopEasy support. It can look up an order, a refund rule, or a shipping option, and it refuses anything outside that job.

---

## Setup

Python 3.14 works. `deepteam` declares `Requires-Python <3.14`, so install with `--ignore-requires-python`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --ignore-requires-python -r requirements.txt
cp .env.example .env
```

Put a Gemini key in `.env`. `.env` is gitignored.

```
GOOGLE_API_KEY=your-gemini-key-here
GEMINI_MODEL_NAME=gemini-3.1-flash-lite
USE_GEMINI_MODEL=true
```

`OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are unused by this demo. `CONFIDENT_API_KEY` is optional: if it is set, DeepEval streams traces to the Confident AI dashboard with no code change. You can also run `deepeval login`.

The free tier for `gemini-3.1-flash-lite` allows 15 generate requests per minute. `llm.py` spaces calls 4.5 seconds apart. Run the eval scripts one at a time so they stay under that limit. Embeddings use `gemini-embedding-001`.

---

## Project layout

| File | What it does |
|---|---|
| `agent_plain.py` | Baseline LangChain agent. Two tools, no tracing, no guardrails. |
| `agent_instrumented.py` | Same agent, plus DeepEval tracing, guardrails, and one MCP tool. Every agent eval imports this. |
| `mcp_server.py` | Local MCP server. Exposes `get_shipping_options` over stdio. Loaded when `agent_instrumented.py` is imported. |
| `rag_agent.py` | Same support role, backed by 9 policy documents in an in-memory vector store. One tool: `search_policies`. |
| `chatbot.py` | Multi-turn chatbot. Uses Gemini's OpenAI-compatible endpoint and keeps its own tool-call loop. |
| `guardrails.py` | DeepTeam input and output guards. Wired into the instrumented agent, the RAG agent, and the chatbot. |
| `llm.py` | Shared Gemini chat model, judge, embeddings, and request pacing. |
| `policies.txt` | Source document for the synthesizer eval. Shipping prices here match `mcp_server.py`. |
| `evals/` | Eval scripts. Run them as programs from the repo root. They are not pytest tests. |
| `evals/INTRO_WHY_AI_EVALS.md` | Short note on why these evals exist. |

---

## Agents

All three agents answer ShopEasy questions. Orders live in memory:

| Order | Status | ETA | Category |
|---|---|---|---|
| `ORD-1042` | Shipped | 2026-05-13 | Electronics |
| `ORD-2099` | Delivered | 2026-05-08 | Clothing |
| `ORD-7777` | Processing | 2026-05-15 | Food |

`agent_plain.py` and `agent_instrumented.py` share two tools:

- `get_order_status(order_id)` — status and ETA
- `get_refund_policy(category)` — electronics, clothing, or food

`agent_instrumented.py` also loads `get_shipping_options` from `mcp_server.py` at import time. The MCP client starts on its own event loop so it does not collide with DeepEval's loop. The agent itself is called with sync `invoke()`, because `ainvoke()` puts LangChain callbacks on a second loop and component evals then crash.

`rag_agent.py` does not use those tools. It embeds the nine policy strings in `POLICY_DOCS` and retrieves the top three chunks. Those strings are a separate knowledge base from `policies.txt`: express shipping in the RAG docs is $15 and standard shipping is 5–7 days. The RAG goldens expect that text.

`chatbot.py` uses the same order and refund tools as the plain agent. It does not load the MCP server.

Sanity-check an agent on its own:

```bash
python agent_plain.py
python agent_instrumented.py
python rag_agent.py
python chatbot.py          # type quit to exit
python mcp_server.py       # blocks; the instrumented agent starts this itself
```

---

## Tracing

`agent_instrumented.py` and `rag_agent.py` share the same DeepEval hooks:

1. `CallbackHandler()` captures every LangChain LLM and tool span.
2. `config={"callbacks": [deepeval_callback]}` on `invoke()` wires that handler in.
3. `update_current_trace(...)` sets `output`, and when a golden has them, `expected_output`, `expected_tools`, and `retrieval_context`.

DeepEval reads those fields from the trace, not straight from the golden, so the eval scripts copy them inside an `@observe` wrapper with `get_current_golden()`.

The chatbot has no LangChain callback. `evals/test_chatbot.py` and `evals/test_chatbot_customreqd.py` build `ConversationalTestCase` turns from the live `chat()` replies and pass them to `evaluate()`.

---

## Guardrails

`guardrails.py` runs on every turn of the instrumented agent, the RAG agent, and the chatbot. The judge is the same Gemini model as the evals. A blocked turn returns:

`I can only help with ShopEasy orders, refunds, and shipping.`

| Guard | Where it runs | What it blocks |
|---|---|---|
| `PromptInjectionGuard` | Input | Instructions that try to override the agent |
| `TopicalGuard` | Input only | Anything outside ShopEasy orders, delivery, refunds, returns, and shipping |
| `ToxicityGuard` | Output | Toxic replies |
| `PrivacyGuard` | Output | Replies that leak private data |
| `IllegalGuard` | Output | Replies that help with illegal requests |

Only a verdict of `unsafe` is refused. `borderline` and `uncertain` are allowed through, so a normal order question is not blocked. `TopicalGuard` is not used on outputs: its output prompt treats the word "policy" as unsafe and would refuse legitimate refund answers.

`evals/test_red_team.py` attacks this guarded agent. DeepTeam generates one attack per vulnerability type, using prompt injection and roleplay, for bias, toxicity, PII leakage, and prompt leakage. `max_concurrent` is 1. Run it by itself. It makes many model calls.

---

## Evals

Run every script from the repo root:

```bash
python evals/test_TaskCompletion.py
python evals/test_TracingComponentsTest.py
python evals/test_multipleEvalsTest.py
python evals/test_customMetricEvals.py
python evals/test_rag_agent.py
python evals/test_chatbot.py
python evals/test_chatbot_customreqd.py
python evals/test_agent_synthesized.py
python evals/test_red_team.py
```

| Script | Agent | Metrics | Cases |
|---|---|---|---|
| `evals/test_TaskCompletion.py` | `support_agent` | Task Completion (0.7) | One order-status question, scored with `evaluate()` |
| `evals/test_TracingComponentsTest.py` | `support_agent` | Task Completion (0.7), Tool Correctness | Order status must call `get_order_status`. Electronics refund must call `get_refund_policy`. |
| `evals/test_multipleEvalsTest.py` | `support_agent` | Answer Relevancy (0.7), Step Efficiency (0.5), Prompt Alignment (0.7) | Three inputs. No expected output. |
| `evals/test_customMetricEvals.py` | `support_agent` | GEval Correctness (0.79) | ORD-1042 is shipped and arrives by May 13th. |
| `evals/test_rag_agent.py` | `rag_support_agent` | Contextual Precision, Contextual Recall, Answer Relevancy, Faithfulness (all 0.7) | Electronics returns, and express shipping at $15. |
| `evals/test_chatbot.py` | `chat` | Turn Relevancy, Knowledge Retention, Conversation Completeness (all 0.5) | Four turns about ORD-1042, its ETA, and an express upgrade. |
| `evals/test_chatbot_customreqd.py` | `chat` | Conversational GEval Correctness (0.80) | Same four turns. Scores role and content. |
| `evals/test_agent_synthesized.py` | `support_agent` | Bias, Toxicity, PII Leakage (all 0.5) | Goldens are generated from `policies.txt` (up to 2 per chunk). Needs `chromadb`. |
| `evals/test_red_team.py` | `support_agent` | DeepTeam attack pass/fail | Bias, toxicity, PII leakage, prompt leakage, each attacked with prompt injection and roleplay. |

Tool Correctness matches tool names, not arguments. The chatbot custom metric does not score `tools_called`.

Component evals (`test_TracingComponentsTest.py`, `test_multipleEvalsTest.py`, `test_customMetricEvals.py`, the RAG eval, and the synthesized eval) use `evals_iterator()`. The task-completion script and both chatbot scripts use `evaluate()`.

---

## Metric reference

| Metric | Needs `expected_output`? | Needs `retrieval_context`? | LLM judge? |
|---|---|---|---|
| Task Completion | No | No | Yes |
| Tool Correctness | No | No | No (name match) |
| Answer Relevancy | No | No | Yes |
| Prompt Alignment | No | No | Yes |
| Step Efficiency | No | No | Yes |
| GEval Correctness | Yes | No | Yes |
| Faithfulness | No | Yes | Yes |
| Contextual Precision | Yes | Yes | Yes |
| Contextual Recall | Yes | Yes | Yes |
| Turn Relevancy | No | No | Yes |
| Knowledge Retention | No | No | Yes |
| Conversation Completeness | No | No | Yes |
| Conversational GEval | No | No | Yes |
| Bias | No | No | Yes |
| Toxicity | No | No | Yes |
| PII Leakage | No | No | Yes |

---

## Notes

- `llm.judge()` returns a `GeminiModel` instance. Passing the instance is what makes DeepEval call Gemini. A bare model-name string is only routed to Gemini when `USE_GEMINI_MODEL=true` and `USE_OPENAI_MODEL` is not set.
- If Gemini returns a schema error that mentions `additional_properties`, the judge retries that call without a response schema and parses the JSON itself.
- `test_agent_synthesized.py` is the slow one: it embeds `policies.txt`, asks Gemini to write goldens, then runs the agent and three safety metrics on each golden.
- Do not run two eval processes at once. They share the 15 request-per-minute quota, and the SDK retries make a 429 last much longer.
