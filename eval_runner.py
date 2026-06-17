import os
from typing import Any, Dict, List, Optional

# DeepEval reads these at import time. Its default per-task timeout is 180s
# (3 min), which is too short for slow cloud judges. Raise the budget.
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1200")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
try:
    from ollama import AsyncClient, Client  # type: ignore
except ModuleNotFoundError:
    # Fallback: minimal Ollama client using httpx, so the API can run even when
    # the `ollama` package isn't installed in the active environment.
    import httpx

    _DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434").rstrip("/")

    class Client:  # type: ignore
        def __init__(self, *, timeout: Optional[float] = None, base_url: Optional[str] = None, **_kwargs):
            self._base_url = (base_url or _DEFAULT_OLLAMA_BASE_URL).rstrip("/")
            self._client = httpx.Client(timeout=timeout)

        def chat(self, *, model: str, messages: List[Dict[str, Any]], options: Optional[Dict[str, Any]] = None):
            # Ollama defaults to streaming NDJSON; disable streaming so we get
            # a single JSON object response (compatible with response.json()).
            payload = {"model": model, "messages": messages, "stream": False}
            if options is not None:
                payload["options"] = options
            r = self._client.post(f"{self._base_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()

    class AsyncClient:  # type: ignore
        def __init__(self, *, timeout: Optional[float] = None, base_url: Optional[str] = None, **_kwargs):
            self._base_url = (base_url or _DEFAULT_OLLAMA_BASE_URL).rstrip("/")
            self._client = httpx.AsyncClient(timeout=timeout)

        async def chat(
            self, *, model: str, messages: List[Dict[str, Any]], options: Optional[Dict[str, Any]] = None
        ):
            payload = {"model": model, "messages": messages, "stream": False}
            if options is not None:
                payload["options"] = options
            r = await self._client.post(f"{self._base_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()

# Keep this file separate from eval_test.py so the API can import it safely.
# eval_test.py runs evaluation at import time; this module must not.

# JUDGE_MODEL = "glm-4.6:cloud"
JUDGE_MODEL = "gpt-oss:120b-cloud"

_OLLAMA_HTTP_TIMEOUT = 1200

_OLLAMA_OPTIONS = {
    "temperature": 0,
    "top_p": 1,
    "seed": 42,
}

_sync_client = Client(timeout=_OLLAMA_HTTP_TIMEOUT)
_async_client = AsyncClient(timeout=_OLLAMA_HTTP_TIMEOUT)


class OllamaCloudModel(DeepEvalBaseLLM):
    def load_model(self):
        return JUDGE_MODEL

    def generate(self, prompt: str) -> str:
        response = _sync_client.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=_OLLAMA_OPTIONS,
        )
        return response["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        response = await _async_client.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=_OLLAMA_OPTIONS,
        )
        return response["message"]["content"]

    def get_model_name(self):
        return JUDGE_MODEL


_judge_model: Optional[OllamaCloudModel] = None
_metrics = None


def _ensure_metrics():
    global _judge_model, _metrics
    if _metrics is not None:
        return
    _judge_model = OllamaCloudModel()
    metric_kwargs = {
        "model": _judge_model,
        "threshold": 0.7,
        "async_mode": False,
        "verbose_mode": False,
    }
    guideline_geval = GEval(
        name="Mortgage Guideline Correctness",
        evaluation_steps=[
            "Check whether the actual output directly answers the input question.",
            "Compare the actual output to the expected output; treat minor formatting "
            "differences (spacing, punctuation, currency symbols) as equivalent when "
            "the meaning matches.",
            "If retrieval context is provided, verify the answer is supported by those "
            "passages and does not contradict them.",
            "Penalize missing key facts, wrong amounts or limits, and invented policy "
            "details not present in the expected output or retrieval context.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        **metric_kwargs,
    )
    _metrics = [
        AnswerRelevancyMetric(**metric_kwargs),
        FaithfulnessMetric(**metric_kwargs),
        HallucinationMetric(**metric_kwargs),
        ContextualRelevancyMetric(**metric_kwargs),
        ContextualPrecisionMetric(**metric_kwargs),
        ContextualRecallMetric(**metric_kwargs),
        guideline_geval,
    ]


def _build_test_case(row: Dict[str, Any]) -> LLMTestCase:
    retrieval_context = row.get("retrieval_context") or []
    context = row.get("context") or retrieval_context
    return LLMTestCase(
        input=str(row.get("input") or ""),
        actual_output=str(row.get("actual_output") or ""),
        expected_output=str(row.get("expected_output") or ""),
        retrieval_context=[str(x) for x in retrieval_context if x is not None],
        context=[str(x) for x in context if x is not None],
    )


def run_eval(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run DeepEval for one or more eval-case rows.

    Returns a flat list of metric results in a JSON-friendly shape.
    """
    _ensure_metrics()
    test_cases = [_build_test_case(r) for r in (rows or [])]
    result = evaluate(
        test_cases=test_cases,
        metrics=_metrics,
        async_config=AsyncConfig(run_async=False),
    )
    out: List[Dict[str, Any]] = []
    for test_result in getattr(result, "test_results", []) or []:
        for md in (getattr(test_result, "metrics_data", None) or []) or []:
            out.append(
                {
                    "name": getattr(md, "name", None),
                    "score": getattr(md, "score", None),
                    "threshold": getattr(md, "threshold", None),
                    "success": getattr(md, "success", None),
                    "reason": getattr(md, "reason", None),
                    "error": getattr(md, "error", None),
                }
            )
    return out

