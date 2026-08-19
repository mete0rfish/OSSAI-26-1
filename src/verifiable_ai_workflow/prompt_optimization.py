"""OpenCQA를 DeepEval PromptOptimizer와 GEPA에 연결한다."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

import yaml
from deepeval.dataset import Golden
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.algorithms import GEPA
from deepeval.prompt import Prompt
from deepeval.test_case import LLMTestCase

from .open_cqa_candidates import OpenCQACase
from .providers.litellm_provider import LiteLLMProvider
from .schemas import StructuredAnswer

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?%?")
_TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")


def split_goldens(cases: list[OpenCQACase]) -> dict[str, list[Golden]]:
    if len(cases) != 30:
        raise ValueError("Week 4 PromptOptimizer에는 OpenCQA 30쌍이 필요합니다")
    counts = {"development": 18, "validation": 6, "test": 6}
    output: dict[str, list[Golden]] = {}
    for split, count in counts.items():
        selected = [case for case in cases if case.course_split == split]
        if len(selected) != count:
            raise ValueError(f"OpenCQA {split} split은 {count}쌍이어야 합니다")
        output[split] = [
            Golden(
                name=case.pair_id,
                input=case.question,
                expected_output=case.reference_answer,
                additional_metadata={
                    "split": case.course_split,
                    "sample_id": case.sample_id,
                    "image_path": case.image_path,
                    "image_sha256": case.image_sha256,
                },
            )
            for case in selected
        ]
    return output


def build_selection_source_evidence(
    cases: list[OpenCQACase], splits: dict[str, list[Golden]]
) -> dict:
    identities = {(case.source_split, case.source_revision, case.source_license) for case in cases}
    if len(identities) != 1:
        raise ValueError("PromptOptimizer OpenCQA pair의 source가 하나가 아닙니다")
    source_split, source_revision, source_license = identities.pop()
    canonical_pairs = json.dumps(
        [case.model_dump(mode="json") for case in sorted(cases, key=lambda case: case.pair_id)],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "dataset_sha256": hashlib.sha256(canonical_pairs).hexdigest(),
        "source_split": source_split,
        "source_revision": source_revision,
        "source_license": source_license,
        "split_sample_ids": {
            name: [(golden.additional_metadata or {})["sample_id"] for golden in goldens]
            for name, goldens in splits.items()
        },
    }


def validate_development_goldens(goldens: list[Golden]) -> None:
    if not goldens or any(
        (golden.additional_metadata or {}).get("split") != "development" for golden in goldens
    ):
        raise ValueError("PromptOptimizer에는 development split만 사용할 수 있습니다")


def _f1(expected: set[str], actual: set[str]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    overlap = len(expected & actual)
    precision, recall = overlap / len(actual), overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


class OpenCqaDeterministicMetric(BaseMetric):
    """GEPA가 재현할 수 있도록 숫자와 핵심 token 겹침을 고정 규칙으로 채점한다."""

    async_mode = False
    threshold = 0.8

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        del args, kwargs
        try:
            answer = StructuredAnswer.model_validate_json(test_case.actual_output).answer
        except Exception as exc:
            self.score, self.reason = 0.0, f"JSON 오류: {exc}"
            self.success, self.error = False, None
            return self.score
        expected = test_case.expected_output or ""
        expected_numbers = set(_NUMBER.findall(expected))
        actual_numbers = set(_NUMBER.findall(answer))
        number_score = _f1(expected_numbers, actual_numbers)
        expected_tokens = set(_TOKEN.findall(expected.casefold()))
        actual_tokens = set(_TOKEN.findall(answer.casefold()))
        token_score = _f1(expected_tokens, actual_tokens)
        self.score = round(0.7 * number_score + 0.3 * token_score, 4)
        missing = sorted(expected_numbers - actual_numbers)
        extra = sorted(actual_numbers - expected_numbers)
        self.reason = (
            f"number_f1={number_score:.3f}, token_f1={token_score:.3f}, "
            f"missing={missing}, extra={extra}"
        )
        self.success, self.error = self.score >= self.threshold, None
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "OpenCQA numeric and token agreement"


def build_prompt_optimizer(
    *,
    goldens: list[Golden],
    model_callback: Callable,
    optimizer_model: DeepEvalBaseLLM,
    config_path: str | Path,
) -> PromptOptimizer:
    validate_development_goldens(goldens)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["optimizer"]
    return PromptOptimizer(
        model_callback=model_callback,
        metrics=[OpenCqaDeterministicMetric()],
        optimizer_model=optimizer_model,
        algorithm=GEPA(
            iterations=config["iterations"],
            minibatch_size=config["minibatch_size"],
            pareto_size=config["pareto_size"],
            patience=config["patience"],
            random_seed=config["random_seed"],
            reflection_model=optimizer_model,
            mutation_model=optimizer_model,
        ),
        async_config=AsyncConfig(run_async=False),
    )


class OpenCqaVlmCallback:
    def __init__(self, provider: LiteLLMProvider, project_root: str | Path) -> None:
        self.provider = provider
        self.project_root = Path(project_root)

    def __call__(self, prompt: Prompt, golden: Golden) -> str:
        metadata = golden.additional_metadata or {}
        image_path = self.project_root / metadata["image_path"]
        image_bytes = image_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != metadata["image_sha256"]:
            raise ValueError("OpenCQA 이미지 bytes가 case SHA-256과 다릅니다")
        image = base64.b64encode(image_bytes).decode("ascii")
        mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        instruction = prompt.interpolate(question=golden.input)
        return self.provider.generate(
            str(metadata["sample_id"]),
            [
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": golden.input},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image}"},
                        },
                    ],
                },
            ],
        )


def score_output(metric: OpenCqaDeterministicMetric, golden: Golden, output: str) -> dict:
    case = LLMTestCase(
        input=golden.input,
        actual_output=output,
        expected_output=golden.expected_output,
    )
    score = metric.measure(case)
    metadata = golden.additional_metadata or {}
    return {
        "pair_id": golden.name,
        "sample_id": metadata.get("sample_id"),
        "image_path": metadata.get("image_path"),
        "split": metadata.get("split"),
        "score": score,
        "reason": metric.reason,
    }
