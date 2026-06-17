# DeepEval metrics for RAG app testing

Reference for evaluating a retrieval-augmented generation (RAG) pipeline with [DeepEval](https://docs.confident-ai.com/). Each metric is run against an `LLMTestCase` and scored by a judge LLM (in this repo: `eval_test.py` → `OllamaCloudModel`).

Official docs: [Metrics introduction](https://docs.confident-ai.com/docs/metrics-introduction), [RAG evaluation](https://docs.confident-ai.com/docs/getting-started-rag).

---

## `LLMTestCase` fields used in RAG evals

| Field | Type | Role in RAG |
|-------|------|-------------|
| `input` | `str` | User question / query sent to the app. |
| `actual_output` | `str` | Final answer your RAG pipeline produced. |
| `expected_output` | `str` | Reference (gold) answer for comparison—not required for all metrics. |
| `retrieval_context` | `list[str]` | Chunks your retriever returned (one string per chunk). Evaluates **retrieval + generation** path. |
| `context` | `list[str]` | Ground-truth source passages (known-correct facts). Used when you have a fixed truth set, not necessarily what was retrieved. |

### `retrieval_context` vs `context`

- **`retrieval_context`**: What your RAG system actually retrieved at runtime. Use for faithfulness and contextual metrics.
- **`context`**: Authoritative reference text (e.g. guideline excerpts). Use for **Hallucination** and similar checks against known facts.

They can contain the same text in tests, but DeepEval treats them as different inputs.

---

## RAG metrics (DeepEval)

These are the metrics most commonly used for RAG / Q&A over documents.

| Metric | Class | What it measures | Required `LLMTestCase` fields |
|--------|--------|------------------|--------------------------------|
| **Answer Relevancy** | `AnswerRelevancyMetric` | Is the answer on-topic for the question? (Ignores retrieval quality and ground truth.) | `input`, `actual_output` |
| **Faithfulness** | `FaithfulnessMetric` | Are claims in the answer supported by **retrieved** chunks? (Groundedness / no unsupported claims.) | `input`, `actual_output`, `retrieval_context` |
| **Contextual Precision** | `ContextualPrecisionMetric` | Of retrieved chunks, are the relevant ones ranked higher? (Retrieval ranking quality.) | `input`, `retrieval_context`, `expected_output` |
| **Contextual Recall** | `ContextualRecallMetric` | Do retrieved chunks contain enough information to produce the **expected** answer? | `input`, `retrieval_context`, `expected_output` |
| **Contextual Relevancy** | `ContextualRelevancyMetric` | Are retrieved chunks relevant to the question? | `input`, `retrieval_context` |
| **Hallucination** | `HallucinationMetric` | Does the answer contradict or go beyond **ground-truth** `context`? | `input`, `actual_output`, `context` |
| **GEval** (custom) | `GEval` | Custom rubric you define (e.g. “mortgage guideline correctness”). | Whatever you pass in `evaluation_params` (e.g. `input`, `actual_output`, `expected_output`) |

### What each metric does (short)

1. **Answer Relevancy** — Generation quality vs question only. Good smoke test; does not need retrieval data.
2. **Faithfulness** — Answer ↔ retrieved docs. Core metric for “did the model stick to what we retrieved?”
3. **Contextual Precision** — Retrieval quality: relevant docs should appear before irrelevant ones.
4. **Contextual Recall** — Retrieval completeness: can the retrieved set support the reference answer?
5. **Contextual Relevancy** — Retrieval relevance: are chunks related to the query?
6. **Hallucination** — Answer ↔ known truth in `context` (not the same as faithfulness to retrieval).
7. **GEval** — Flexible LLM-as-judge with your own `criteria` or `evaluation_steps`.

---

## Field requirements by metric (quick lookup)

```
Answer Relevancy:     input + actual_output
Faithfulness:         input + actual_output + retrieval_context
Contextual Precision: input + retrieval_context + expected_output
Contextual Recall:    input + retrieval_context + expected_output
Contextual Relevancy: input + retrieval_context
Hallucination:        input + actual_output + context
GEval:                fields listed in evaluation_params (you choose)
```

---

## Example `LLMTestCase` for a full RAG eval

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="How are NSFs treated on bank statement loans?",
    actual_output="Up to 3 occurrences in the last 12 months if...",  # from your RAG app
    expected_output="We allow up to 3 occurrences...",               # gold / SME answer
    retrieval_context=[                                              # from your retriever
        "NSF limits: up to 3 in 12 months if 1+ in last 3 months...",
        "Overdraft protection may be excluded when...",
    ],
    context=[                                                        # ground-truth guideline text
        "We allow up to 3 occurrences in the most recent 12 months...",
    ],
)
```

---

## Metrics enabled in this repo (`eval_test.py`)

Current configuration:

| Metric | Status |
|--------|--------|
| Answer Relevancy | Enabled |
| Contextual Recall | Enabled |
| Hallucination | Enabled |
| Faithfulness | Not enabled |
| Contextual Precision | Not enabled |
| Contextual Relevancy | Not enabled |
| GEval | Not enabled |

### Data in `mortgageQ.json`

Each row provides:

- `input`, `actual_output`, `expected_output`
- `retrieval_context`: currently `[]` (empty)

Implications:

| Metric | Works with current JSON? |
|--------|---------------------------|
| Answer Relevancy | Yes |
| Contextual Recall | Needs non-empty `retrieval_context` |
| Hallucination | Needs a `context` field (not in JSON today) |
| Faithfulness, Contextual Precision, Contextual Relevancy | Need non-empty `retrieval_context` |

To run retrieval/context metrics, populate chunks when building test cases (e.g. save retrieved passages from `rag_pipeline.py` into the eval dataset).

---

## Wiring multiple metrics

```python
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)

judge_model = ...  # DeepEvalBaseLLM, e.g. OllamaCloudModel
kwargs = {"model": judge_model, "threshold": 0.7}

metrics = [
    AnswerRelevancyMetric(**kwargs),
    FaithfulnessMetric(**kwargs),
    ContextualRecallMetric(**kwargs),
    HallucinationMetric(**kwargs),
]

evaluate(test_cases=test_cases, metrics=metrics)
```

### GEval example (custom RAG rubric)

```python
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

GEval(
    name="Mortgage Guideline Correctness",
    criteria="Score whether the actual output correctly answers the question using the expected output as reference.",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    model=judge_model,
    threshold=0.7,
)
```

---

## Suggested RAG eval workflow

1. **Collect** real queries (`input`) and run your pipeline → `actual_output` + `retrieval_context`.
2. **Add** SME `expected_output` and, for hallucination checks, guideline `context` snippets.
3. **Run** metrics in layers:
   - Retrieval: Contextual Relevancy, Contextual Precision, Contextual Recall
   - Generation: Answer Relevancy, Faithfulness, Hallucination
   - Business rules: GEval with domain `criteria`
4. **Tune** `threshold` per metric (default in DeepEval is `0.5`; this project uses `0.7`).

---

## Run eval in this project

From the repo root (requires Ollama / judge model configured in `eval_test.py`):

```bash
cd /home/galaxy/Python/learning/poc_rag
python3 eval_test.py
```

Results may also sync to Confident AI if the DeepEval CLI/project is linked.

---

## Related (non-RAG) DeepEval metrics

Useful for chat apps but not specific to retrieval:

| Metric | Typical use | Required fields |
|--------|-------------|-----------------|
| `ExactMatchMetric` | String match to reference | `input`, `actual_output`, `expected_output` |
| `SummarizationMetric` | Summary quality | `input`, `actual_output` |
| `BiasMetric`, `ToxicityMetric` | Safety | `input`, `actual_output` |

For RAG document Q&A, prefer the metrics in the first table above.
