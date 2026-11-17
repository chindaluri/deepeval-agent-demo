"""Shared Gemini model for the agents, the DeepEval judge, and the guardrails."""

import json
import os
import re
import threading
import time

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("USE_GEMINI_MODEL", "true")
os.environ.setdefault("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")

from deepeval.models import GeminiModel
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = "gemini-embedding-001"

# The free tier allows 15 generate requests per minute for this model.
_PACE_SECONDS = 4.5
_pace_lock = threading.Lock()
_next_call = 0.0


def pace() -> None:
    """Space Gemini calls so a full eval run stays under the free-tier limit."""
    global _next_call
    with _pace_lock:
        now = time.monotonic()
        wait = _next_call - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _next_call = now + _PACE_SECONDS


def _strip_unsupported_schema(node):
    """Gemini's developer API rejects additionalProperties in response schemas."""
    if isinstance(node, dict):
        node.pop("additionalProperties", None)
        node.pop("additional_properties", None)
        for value in node.values():
            _strip_unsupported_schema(value)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported_schema(item)


def _json_from_text(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group(0) if match else text
    return json.loads(raw)


class PacedGeminiModel(GeminiModel):
    def generate(self, prompt, schema=None, audio=None):
        pace()
        if schema is None:
            return super().generate(prompt, schema=None, audio=audio)
        try:
            return super().generate(prompt, schema=schema, audio=audio)
        except Exception as error:
            if "additional_properties" not in str(error):
                raise
        text, cost = super().generate(prompt, schema=None, audio=audio)
        return schema.model_validate(_json_from_text(text)), cost

    async def a_generate(self, prompt, schema=None, audio=None):
        pace()
        if schema is None:
            return await super().a_generate(prompt, schema=None, audio=audio)
        try:
            return await super().a_generate(prompt, schema=schema, audio=audio)
        except Exception as error:
            if "additional_properties" not in str(error):
                raise
        text, cost = await super().a_generate(prompt, schema=None, audio=audio)
        return schema.model_validate(_json_from_text(text)), cost


def api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY in .env.")
    return key


def _pace_calls(obj, *method_names: str) -> None:
    for name in method_names:
        original = getattr(obj, name)

        def wrapped(*args, _original=original, **kwargs):
            pace()
            return _original(*args, **kwargs)

        object.__setattr__(obj, name, wrapped)


def judge() -> PacedGeminiModel:
    """DeepEval judge. Pass this object anywhere a metric takes model=."""
    return PacedGeminiModel(model=MODEL_NAME, api_key=api_key(), temperature=0)


def chat_model() -> ChatGoogleGenerativeAI:
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
        google_api_key=api_key(),
    )
    _pace_calls(model, "invoke", "ainvoke")
    return model


class GeminiEmbeddingModel(DeepEvalBaseEmbeddingModel):
    """DeepEval embedder backed by the same Gemini embeddings as the RAG agent."""

    def __init__(self):
        super().__init__(EMBEDDING_MODEL)

    def load_model(self):
        return embeddings()

    def embed_text(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        return EMBEDDING_MODEL


def embeddings() -> GoogleGenerativeAIEmbeddings:
    model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key(),
    )
    _pace_calls(model, "embed_query", "embed_documents")
    return model
