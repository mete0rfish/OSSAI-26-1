"""모델 호출을 실행하고 아직 채점하지 않은 관찰값을 남긴다."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..preprocessing import load_document
from ..providers.base import ModelProvider
from ..schemas import EvaluationCase, ModelObservation
from .inputs import build_page_input


def build_messages(
    *,
    question: str,
    page_input: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"질문: {question}"},
                *page_input,
            ],
        },
    ]


def run_cases(
    *,
    cases: list[EvaluationCase],
    prepared_documents: str | Path,
    prompt_path: str | Path,
    provider: ModelProvider,
    on_observation: Callable[[ModelObservation], None] | None = None,
) -> list[ModelObservation]:
    system_prompt = Path(prompt_path).read_text(encoding="utf-8")
    observations: list[ModelObservation] = []
    for case in cases:
        document, manifest_path = load_document(prepared_documents, case.document_id)
        messages = build_messages(
            question=case.question,
            page_input=build_page_input(
                document,
                manifest_path,
                getattr(provider, "max_images_per_prompt", None),
            ),
            system_prompt=system_prompt,
        )
        try:
            raw_output = provider.generate(case.sample_id, messages)
            model_error = None
        except Exception as exc:
            raw_output = None
            model_error = f"{type(exc).__name__}: {exc}"
        observation = ModelObservation(
            sample_id=case.sample_id,
            family_id=case.family_id,
            total_pages=document.total_pages,
            raw_output=raw_output,
            model_error=model_error,
            model_call=getattr(provider, "last_call", None),
            evidence_kind=provider.evidence_kind,
        )
        observations.append(observation)
        if on_observation is not None:
            on_observation(observation)
    return observations
