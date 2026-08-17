"""VLM 응답을 구조, 정답, 답변 보류와 근거 페이지 고정 규칙으로 평가한다."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from pydantic import ValidationError

from ..schemas import EvaluationCase, EvaluationResult, ModelObservation, StructuredAnswer

SCORING_PROFILE = "aihub-vqa-deterministic-v4"
PASS_REQUIREMENTS = (
    "schema_validity",
    "abstention_correct",
    "answer_correct",
    "evidence_coverage",
)
DIAGNOSTIC_METRICS = (
    "json_object_only",
    "answer_exact",
    "answer_similarity",
    "numeric_match",
    "quote_answer_support",
)
SCORE_NAMES = (*PASS_REQUIREMENTS, *DIAGNOSTIC_METRICS, "task_success")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^0-9a-z가-힣%]+", "", normalized)


def _numbers(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", unicodedata.normalize("NFKC", text))


def _similarity(actual: str, expected: str) -> float:
    return SequenceMatcher(None, _normalize(actual), _normalize(expected)).ratio()


def _contains_answer_fact(text: str, answer: str) -> bool:
    answer_numbers = _numbers(answer)
    if answer_numbers:
        return all(number in _numbers(text) for number in answer_numbers)
    normalized_answer = _normalize(answer)
    return bool(normalized_answer and normalized_answer in _normalize(text))


def parse_json_output(raw_output: Any) -> Any:
    if isinstance(raw_output, str):
        text = raw_output.strip()
        if text.startswith("```") and text.endswith("```"):
            first_newline = text.find("\n")
            if first_newline >= 0:
                text = text[first_newline + 1 : -3].strip()
        raw_output = json.loads(text)
    return raw_output


def parse_output(raw_output: Any) -> StructuredAnswer:
    return StructuredAnswer.model_validate(parse_json_output(raw_output))


def score_output(
    raw_output: Any,
    case: EvaluationCase,
) -> tuple[StructuredAnswer | None, dict[str, float], dict[str, str]]:
    try:
        answer = parse_output(raw_output)
        schema_validity = 1.0
        schema_reason = "StructuredAnswer 형식 통과"
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        answer = None
        schema_validity = 0.0
        schema_reason = f"{type(exc).__name__}: {exc}"

    json_object_only = float(
        not isinstance(raw_output, str)
        or (raw_output.strip().startswith("{") and raw_output.strip().endswith("}"))
    )
    actual_answer = answer.answer if answer else ""
    accepted_answers = [case.expected.answer, *case.expected.accepted_answers]
    answer_exact = float(
        bool(answer)
        and answer.abstained == case.expected.abstained
        and _normalize(actual_answer) == _normalize(case.expected.answer)
    )
    answer_similarity = _similarity(actual_answer, case.expected.answer) if answer else 0.0
    expected_numbers = _numbers(case.expected.answer)
    actual_numbers = _numbers(actual_answer)
    numeric_match = float(not expected_numbers or actual_numbers == expected_numbers)
    abstention_correct = float(bool(answer) and answer.abstained == case.expected.abstained)

    answer_correct = float(
        bool(answer)
        and answer.abstained == case.expected.abstained
        and _normalize(actual_answer) in {_normalize(value) for value in accepted_answers}
    )

    actual_pages = {item.page_number for item in answer.evidence} if answer else set()
    expected_pages = set(case.expected.pages)
    evidence_coverage = float(
        bool(actual_pages & expected_pages) if expected_pages else not actual_pages
    )

    quote_answer_scores = (
        [float(_contains_answer_fact(item.quote, actual_answer)) for item in answer.evidence]
        if answer and not answer.abstained
        else [1.0]
        if answer and answer.abstained
        else []
    )
    quote_answer_support = (
        sum(quote_answer_scores) / len(quote_answer_scores) if quote_answer_scores else 0.0
    )

    required_scores = {
        "schema_validity": schema_validity,
        "abstention_correct": abstention_correct,
        "answer_correct": answer_correct,
        "evidence_coverage": evidence_coverage,
    }
    task_success = float(all(required_scores[name] == 1.0 for name in PASS_REQUIREMENTS))
    scores = {
        "json_object_only": json_object_only,
        "schema_validity": schema_validity,
        "answer_exact": answer_exact,
        "answer_similarity": round(answer_similarity, 4),
        "numeric_match": numeric_match,
        "abstention_correct": abstention_correct,
        "answer_correct": answer_correct,
        "evidence_coverage": evidence_coverage,
        "quote_answer_support": round(quote_answer_support, 4),
        "task_success": task_success,
    }
    reasons = {
        "json_object_only": (
            "JSON object만 반환"
            if json_object_only
            else "Markdown fence 또는 부가 텍스트가 있어 정리 후 파싱"
        ),
        "schema_validity": schema_reason,
        "answer_exact": f"실제 답={actual_answer!r}, 기준 답={case.expected.answer!r}",
        "answer_similarity": f"정규화 문자 유사도={answer_similarity:.4f}",
        "numeric_match": f"실제 숫자={actual_numbers}, 기대 숫자={expected_numbers}",
        "abstention_correct": (
            f"실제 보류={answer.abstained if answer else None}, "
            f"기대 보류={case.expected.abstained}"
        ),
        "answer_correct": (
            "기준 답 또는 명시 허용답과 일치"
            if answer_correct
            else "기준 답과 명시 허용답에 일치하지 않음"
        ),
        "evidence_coverage": (
            "가능한 근거 페이지를 하나 이상 인용"
            if evidence_coverage
            else "가능한 근거 페이지를 인용하지 않음"
        ),
        "quote_answer_support": (
            f"모델이 작성한 인용문 내 답 핵심값 포함 비율={quote_answer_support:.4f}; "
            "이미지 근거 일치 여부를 증명하는 점수는 아님"
        ),
        "task_success": "모든 필수 정량 기준 통과" if task_success else "하나 이상 실패",
    }
    return answer, scores, reasons


def score_observations(
    cases: list[EvaluationCase],
    observations: list[ModelObservation],
) -> list[EvaluationResult]:
    case_by_id = {case.sample_id: case for case in cases}
    results: list[EvaluationResult] = []
    for observation in observations:
        case = case_by_id[observation.sample_id]
        if observation.model_error:
            scores = dict.fromkeys(SCORE_NAMES, 0.0)
            results.append(
                EvaluationResult(
                    sample_id=case.sample_id,
                    family_id=case.family_id,
                    status="inconclusive",
                    raw_output={"error": observation.model_error},
                    scores=scores,
                    reasons={name: observation.model_error for name in scores},
                    evidence_kind=observation.evidence_kind,
                    evaluation_mode=observation.evaluation_mode,
                    provider_status="provider_error",
                    model_call=observation.model_call,
                    route_attempts=observation.route_attempts,
                )
            )
            continue

        output, scores, reasons = score_output(observation.raw_output, case)
        if output and any(
            evidence.page_number > observation.total_pages for evidence in output.evidence
        ):
            scores["evidence_coverage"] = 0.0
            scores["task_success"] = 0.0
            reasons["evidence_coverage"] = "근거 페이지가 문서 페이지 범위를 벗어났습니다"
            reasons["task_success"] = "근거 페이지가 문서 페이지 범위를 벗어남"
        status = "passed" if scores["task_success"] == 1.0 else "failed"
        results.append(
            EvaluationResult(
                sample_id=case.sample_id,
                family_id=case.family_id,
                status=status,
                output=output,
                raw_output=observation.raw_output,
                scores=scores,
                reasons=reasons,
                evidence_kind=observation.evidence_kind,
                evaluation_mode=observation.evaluation_mode,
                provider_status=(
                    "invalid_output"
                    if scores["schema_validity"] == 0.0
                    else observation.provider_status
                ),
                model_call=observation.model_call,
                route_attempts=observation.route_attempts,
            )
        )
    return results
