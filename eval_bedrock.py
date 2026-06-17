import asyncio
import json
import os
from pathlib import Path

import boto3

# DeepEval reads these at import time. Its default per-task timeout is 180s
# (3 min), which is too short for slow cloud judges and triggers
# asyncio.TimeoutError. Raise the budget (or set DEEPEVAL_DISABLE_TIMEOUTS=1).
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1200")

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
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

# Bedrock model ID or inference profile ID (region-specific).
JUDGE_MODEL = os.environ.get(
    "BEDROCK_JUDGE_MODEL_ID",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
FILE_NAME = "mortgageQ.json"

_BEDROCK_INFERENCE_CONFIG = {
    "temperature": 0,
    "topP": 1,
}

_bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


class BedrockModel(DeepEvalBaseLLM):

    def load_model(self):
        return JUDGE_MODEL

    def generate(self, prompt: str) -> str:
        response = _bedrock_client.converse(
            modelId=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig=_BEDROCK_INFERENCE_CONFIG,
        )
        return response["output"]["message"]["content"][0]["text"]

    async def a_generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self):
        return JUDGE_MODEL


judge_model = BedrockModel()

_metric_kwargs = {
    "model": judge_model,
    "threshold": 0.7,
    "async_mode": False,
    "verbose_mode": True,
}

# Custom guideline correctness metric
_guideline_geval = GEval(
    name="Mortgage Guideline Correctness",
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

# Sync mode runs one metric at a time (avoids Bedrock throttling).
_result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),
)

_print_metric_scores(_result)
