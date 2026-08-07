"""VLM 응답을 구조, 정답, 답변 보류와 근거 페이지 고정 규칙으로 평가한다."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from pydantic import ValidationError

from ..schemas import EvaluationCase, EvaluationResult, ModelObservation, StructuredAnswer

SCORING_PROFILE = "aihub-vqa-deterministic-v2"
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
    "answer_anls",
    "answer_token_f1",
    "numeric_match",
    "evidence_page_precision",
    "evidence_page_recall",
    "evidence_page_f1",
    "quote_answer_support",
)
SCORE_NAMES = (*PASS_REQUIREMENTS, *DIAGNOSTIC_METRICS, "task_success")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^0-9a-z가-힣]+", "", normalized)


def _numbers(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", unicodedata.normalize("NFKC", text))


def _similarity(actual: str, expected: str) -> float:
    return SequenceMatcher(None, _normalize(actual), _normalize(expected)).ratio()


def _edit_similarity(actual: str, expected: str) -> float:
    """DocVQA ANLS 계산에 쓰는 정규화 Levenshtein similarity."""
    left = _normalize(actual)
    right = _normalize(expected)
    if not left or not right:
        return float(left == right)

    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_character != right_character),
                )
            )
        previous = current
    return 1 - previous[-1] / max(len(left), len(right))


def _answer_anls(actual: str, expected: str) -> float:
    similarity = _edit_similarity(actual, expected)
    return similarity if similarity >= 0.5 else 0.0


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"-?\d+(?:\.\d+)?|[a-z]+|[가-힣]+", normalized)


def _token_f1(actual: str, expected: str) -> float:
    actual_counts: dict[str, int] = {}
    expected_counts: dict[str, int] = {}
    for token in _tokens(actual):
        actual_counts[token] = actual_counts.get(token, 0) + 1
    for token in _tokens(expected):
        expected_counts[token] = expected_counts.get(token, 0) + 1
    if not actual_counts or not expected_counts:
        return float(actual_counts == expected_counts)
    overlap = sum(
        min(count, expected_counts.get(token, 0)) for token, count in actual_counts.items()
    )
    precision = overlap / sum(actual_counts.values())
    recall = overlap / sum(expected_counts.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _contains_answer_fact(text: str, answer: str) -> bool:
    answer_numbers = _numbers(answer)
    if answer_numbers:
        return all(number in _numbers(text) for number in answer_numbers)
    normalized_answer = _normalize(answer)
    return bool(normalized_answer and normalized_answer in _normalize(text))


def parse_output(raw_output: Any) -> StructuredAnswer:
    if isinstance(raw_output, str):
        text = raw_output.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
        raw_output = json.loads(text)
    return StructuredAnswer.model_validate(raw_output)


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
    answer_exact = float(
        bool(answer)
        and answer.abstained == case.expected.abstained
        and _normalize(actual_answer) == _normalize(case.expected.answer)
    )
    answer_similarity = _similarity(actual_answer, case.expected.answer) if answer else 0.0
    answer_anls = _answer_anls(actual_answer, case.expected.answer) if answer else 0.0
    answer_token_f1 = _token_f1(actual_answer, case.expected.answer) if answer else 0.0
    expected_numbers = _numbers(case.expected.answer)
    actual_numbers = _numbers(actual_answer)
    numeric_match = float(not expected_numbers or actual_numbers == expected_numbers)
    abstention_correct = float(bool(answer) and answer.abstained == case.expected.abstained)

    expected_text_without_numbers = re.sub(
        r"-?\d+(?:\.\d+)?",
        "",
        _normalize(case.expected.answer),
    )
    if case.expected.abstained:
        answer_correct = answer_exact
    elif expected_numbers and len(expected_text_without_numbers) <= 3:
        answer_correct = float(bool(answer) and numeric_match == 1.0)
    elif expected_numbers:
        answer_correct = float(bool(answer) and numeric_match == 1.0 and answer_similarity >= 0.65)
    else:
        answer_correct = float(bool(answer) and answer_similarity >= 0.75)

    actual_pages = {item.page_number for item in answer.evidence} if answer else set()
    expected_pages = set(case.expected.pages)
    page_hits = len(actual_pages & expected_pages)
    page_precision = page_hits / len(actual_pages) if actual_pages else float(not expected_pages)
    page_recall = page_hits / len(expected_pages) if expected_pages else float(not actual_pages)
    page_f1 = (
        2 * page_precision * page_recall / (page_precision + page_recall)
        if page_precision + page_recall
        else 0.0
    )
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
        "answer_anls": round(answer_anls, 4),
        "answer_token_f1": round(answer_token_f1, 4),
        "numeric_match": numeric_match,
        "abstention_correct": abstention_correct,
        "answer_correct": answer_correct,
        "evidence_page_precision": round(page_precision, 4),
        "evidence_page_recall": round(page_recall, 4),
        "evidence_page_f1": round(page_f1, 4),
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
        "answer_exact": f"실제 답={actual_answer!r}, 기대 답={case.expected.answer!r}",
        "answer_similarity": f"정규화 문자 유사도={answer_similarity:.4f}",
        "answer_anls": f"DocVQA ANLS={answer_anls:.4f}",
        "answer_token_f1": f"답 토큰 F1={answer_token_f1:.4f}",
        "numeric_match": f"실제 숫자={actual_numbers}, 기대 숫자={expected_numbers}",
        "abstention_correct": (
            f"실제 보류={answer.abstained if answer else None}, "
            f"기대 보류={case.expected.abstained}"
        ),
        "answer_correct": "정답 허용 기준 통과" if answer_correct else "정답 허용 기준 실패",
        "evidence_page_precision": (
            f"실제 페이지={sorted(actual_pages)}, 기대 페이지={sorted(expected_pages)}"
        ),
        "evidence_page_recall": (
            f"실제 페이지={sorted(actual_pages)}, 기대 페이지={sorted(expected_pages)}"
        ),
        "evidence_page_f1": f"페이지 F1={page_f1:.4f}",
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
