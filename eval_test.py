import json
import os
from pathlib import Path

# DeepEval reads these at import time. Its default per-task timeout is 180s
# (3 min), which is too short for slow cloud judges like glm-4.7:cloud /
# gpt-oss:120b-cloud and triggers asyncio.TimeoutError. Raise the budget
# (or set DEEPEVAL_DISABLE_TIMEOUTS=1 to remove the limit entirely).
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1200")

try:
    # Load local .env so OPENAI_API_KEY is available at runtime.
    # If python-dotenv isn't installed, we fall back to normal env vars.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    # ContextualRelevancyMetric,
    # FaithfulnessMetric,
    GEval,
    # HallucinationMetric,
    # ContextualPrecisionMetric,
    # ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import DeepEvalBaseLLM
from ollama import Client, AsyncClient

# Select which judge backend to use: "ollama" (default) or "openai".
# Set these explicitly in code (not via environment variables).
JUDGE_PROVIDER = "openai"
OPENAI_JUDGE_MODEL = "gpt-5.5"

JUDGE_MODEL = "gpt-oss:120b-cloud"
FILE_NAME = "mq-Deephaven.json"

# Generous HTTP timeout for the cloud judge; None disables httpx's read
# timeout so a single slow response doesn't blow up the run.
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
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options=_OLLAMA_OPTIONS,
        )

        return response["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        response = await _async_client.chat(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options=_OLLAMA_OPTIONS,
        )

        return response["message"]["content"]

    def get_model_name(self):
        return JUDGE_MODEL


class OpenAIJudgeModel(DeepEvalBaseLLM):

    def load_model(self):
        return OPENAI_JUDGE_MODEL

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI judge selected (JUDGE_PROVIDER=openai) but the 'openai' "
                "package is not installed. Install it (e.g. `pip install openai`) "
                "or set JUDGE_PROVIDER=ollama."
            ) from e

        client = OpenAI()
        resp = client.chat.completions.create(
            model=OPENAI_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            # temperature=0,
        )
        return resp.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI judge selected (JUDGE_PROVIDER=openai) but the 'openai' "
                "package is not installed. Install it (e.g. `pip install openai`) "
                "or set JUDGE_PROVIDER=ollama."
            ) from e

        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=OPENAI_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            # temperature=0,
        )
        return resp.choices[0].message.content or ""

    def get_model_name(self):
        return OPENAI_JUDGE_MODEL


if JUDGE_PROVIDER == "openai":
    judge_model = OpenAIJudgeModel()
elif JUDGE_PROVIDER == "ollama":
    judge_model = OllamaCloudModel()
else:
    raise ValueError(
        f"Unknown JUDGE_PROVIDER={JUDGE_PROVIDER!r}. Use 'ollama' or 'openai'."
    )

_metric_kwargs = {
    "model": judge_model,
    "threshold": 0.7,
    "async_mode": False,
    "verbose_mode": True,
}

# Custom guideline correctness metric
_guideline_geval = GEval(
    name="MQG Correctness",
    evaluation_steps=[
        "Check whether the actual output directly answers the input question.",
        "Compare the actual output to the expected output; treat minor formatting "
        "differences (spacing, punctuation, currency symbols) as equivalent when "
        "the meaning matches.",
        # "If retrieval context is provided, verify the answer is supported by those "
        # "passages and does not contradict them.",
        "Penalize missing key facts, wrong amounts or limits, and invented policy "
        # "details not present in the expected output or retrieval context.",
        "details not present in the expected output.",
    ],
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        # SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    **_metric_kwargs,
)

metrics = [
    AnswerRelevancyMetric(**_metric_kwargs),
    # FaithfulnessMetric(**_metric_kwargs),
    # HallucinationMetric(**_metric_kwargs),
    # ContextualRelevancyMetric(**_metric_kwargs),
    # ContextualPrecisionMetric(**_metric_kwargs),
    # ContextualRecallMetric(**_metric_kwargs),
    _guideline_geval,
]


def _build_test_case(row: dict) -> LLMTestCase:
    retrieval_context = row.get("retrieval_context") or []
    # Hallucination uses ground-truth `context`; fall back to retrieved chunks.
    context = row.get("context") or retrieval_context
    return LLMTestCase(
        input=row["input"],
        actual_output=row["actual_output"],
        expected_output=row.get("expected_output") or "",
        retrieval_context=retrieval_context,
        context=context,
    )


def _print_metric_scores(result) -> None:
    for i, test_result in enumerate(result.test_results, start=1):
        print(f"\n{'=' * 60}")
        print(f"Test case {i}: {test_result.input!r}")
        print(f"{'=' * 60}")
        if not test_result.metrics_data:
            print("  (no metric results)")
            continue
        for metric_data in test_result.metrics_data:
            label = metric_data.name
            score = metric_data.score
            score_str = f"{score:.4f}" if score is not None else "n/a"
            status = "PASS" if metric_data.success else "FAIL"
            print(
                f"  {label:<22} score={score_str}  "
                f"threshold={metric_data.threshold}  [{status}]"
            )
            if metric_data.reason:
                print(f"    reason: {metric_data.reason}")
            if metric_data.error:
                print(f"    error: {metric_data.error}")

_eval_data_path = Path(__file__).resolve().parent / FILE_NAME
with _eval_data_path.open(encoding="utf-8") as f:
    _loaded = json.load(f)

# FILE_NAME is a list; FILE_NAME (and similar exports) may be one object.
if isinstance(_loaded, dict):
    _mortgage_rows = [_loaded]
elif isinstance(_loaded, list):
    _mortgage_rows = _loaded
else:
    raise TypeError(
        f"{_eval_data_path.name} must be a JSON object or array of objects, "
        f"got {type(_loaded).__name__}"
    )

test_cases = [_build_test_case(row) for row in _mortgage_rows]

# Default evaluate() uses async with max_concurrent=20, which overwhelms
# Ollama Cloud and returns 429. Sync mode runs one metric at a time.
_result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),
)

_print_metric_scores(_result)