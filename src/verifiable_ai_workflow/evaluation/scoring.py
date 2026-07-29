"""모델 응답을 정답, 숫자, 근거 페이지와 실제 PDF 문장으로 평가한다."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..preprocessing import load_document
from ..schemas import EvaluationCase, EvaluationResult, ModelObservation, StructuredAnswer


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
                    previous[right_index - 1]
                    + int(left_character != right_character),
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
        min(count, expected_counts.get(token, 0))
        for token, count in actual_counts.items()
    )
    precision = overlap / sum(actual_counts.values())
    recall = overlap / sum(expected_counts.values())
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _best_quote_similarity(quote: str, page_text: str) -> float:
    needle = _normalize(quote)
    haystack = _normalize(page_text)
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 1.0

    window_size = min(len(haystack), max(len(needle), round(len(needle) * 1.4)))
    step = max(1, len(needle) // 4)
    return max(
        SequenceMatcher(None, needle, haystack[start : start + window_size]).ratio()
        for start in range(0, max(1, len(haystack) - window_size + 1), step)
    )


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


def load_page_texts(
    prepared_documents: str | Path,
    document_id: str,
) -> dict[int, str]:
    document, manifest_path = load_document(prepared_documents, document_id)
    return {
        page.page_number: (manifest_path.parent / page.text_path).read_text(encoding="utf-8")
        for page in document.pages
    }


def score_output(
    raw_output: Any,
    case: EvaluationCase,
    *,
    page_texts: dict[int, str] | None = None,
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
        bool(actual_pages & expected_pages)
        if expected_pages
        else not actual_pages
    )

    quote_scores: list[float] = []
    quote_answer_scores: list[float] = []
    verifiable_quote_scores: list[float] = []
    if answer and not answer.abstained and page_texts is not None:
        for evidence in answer.evidence:
            page_text = page_texts.get(evidence.page_number, "")
            text_similarity = _best_quote_similarity(evidence.quote, page_text)
            answer_in_quote = _contains_answer_fact(evidence.quote, actual_answer)
            answer_in_page = _contains_answer_fact(page_text, actual_answer)
            fact_support = float(answer_in_quote and answer_in_page)
            score = max(text_similarity, fact_support)
            quote_scores.append(score)
            quote_answer_scores.append(float(answer_in_quote))
            if _contains_answer_fact(page_text, case.expected.answer):
                verifiable_quote_scores.append(score)
    elif answer and answer.abstained:
        quote_scores = [1.0]
        quote_answer_scores = [1.0]
        verifiable_quote_scores = [1.0]

    quote_answer_support = (
        sum(quote_answer_scores) / len(quote_answer_scores)
        if quote_answer_scores
        else 0.0
    )
    quote_verifiability = (
        len(verifiable_quote_scores) / len(quote_scores)
        if quote_scores
        else float(bool(answer and answer.abstained))
    )
    quote_grounding = (
        sum(quote_scores) / len(quote_scores)
        if quote_scores
        else float(bool(answer and answer.abstained))
    )
    verifiable_grounding = (
        sum(verifiable_quote_scores) / len(verifiable_quote_scores)
        if verifiable_quote_scores
        else None
    )
    quote_required = (
        page_texts is not None
        and not case.expected.abstained
        and verifiable_grounding is not None
    )
    quote_passed = (
        not quote_required
        or verifiable_grounding is not None
        and verifiable_grounding >= 0.8
    )

    task_success = float(
        schema_validity == 1.0
        and abstention_correct == 1.0
        and answer_correct == 1.0
        and evidence_coverage == 1.0
        and quote_passed
    )
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
        "quote_verifiability": round(quote_verifiability, 4),
        "quote_grounding": round(quote_grounding, 4),
        "task_success": task_success,
    }
    reasons = {
        "json_object_only": (
            "JSON object만 반환"
            if json_object_only
            else "Markdown fence 또는 부가 텍스트가 있어 정리 후 파싱"
        ),
        "schema_validity": schema_reason,
        "answer_exact": f"actual={actual_answer!r}, expected={case.expected.answer!r}",
        "answer_similarity": f"normalized character similarity={answer_similarity:.4f}",
        "answer_anls": f"DocVQA ANLS={answer_anls:.4f}",
        "answer_token_f1": f"answer token F1={answer_token_f1:.4f}",
        "numeric_match": (f"actual_numbers={actual_numbers}, expected_numbers={expected_numbers}"),
        "abstention_correct": (
            f"actual={answer.abstained if answer else None}, expected={case.expected.abstained}"
        ),
        "answer_correct": "정답 허용 기준 통과" if answer_correct else "정답 허용 기준 실패",
        "evidence_page_precision": (
            f"actual_pages={sorted(actual_pages)}, expected_pages={sorted(expected_pages)}"
        ),
        "evidence_page_recall": (
            f"actual_pages={sorted(actual_pages)}, expected_pages={sorted(expected_pages)}"
        ),
        "evidence_page_f1": f"page F1={page_f1:.4f}",
        "evidence_coverage": (
            "가능한 근거 페이지를 하나 이상 인용"
            if evidence_coverage
            else "가능한 근거 페이지를 인용하지 않음"
        ),
        "quote_answer_support": (
            f"인용문 내 답 핵심값 포함 비율={quote_answer_support:.4f}"
        ),
        "quote_verifiability": (
            f"PDF 추출 텍스트로 검증 가능한 인용 비율={quote_verifiability:.4f}"
        ),
        "quote_grounding": (
            (
                f"PDF page text/fact support={quote_grounding:.4f}; "
                f"gate_score={verifiable_grounding:.4f}"
                if verifiable_grounding is not None
                else (
                    f"PDF page text/fact support={quote_grounding:.4f}; "
                    "표·차트 값이 추출되지 않아 gate에서 제외"
                )
            )
            if page_texts is not None
            else "PDF page text를 제공하지 않아 gate에서 제외"
        ),
        "task_success": "모든 필수 정량 기준 통과" if task_success else "하나 이상 실패",
    }
    return answer, scores, reasons


def score_observations(
    cases: list[EvaluationCase],
    observations: list[ModelObservation],
    *,
    prepared_documents: str | Path | None = None,
) -> list[EvaluationResult]:
    case_by_id = {case.sample_id: case for case in cases}
    text_cache: dict[str, dict[int, str]] = {}
    results: list[EvaluationResult] = []
    for observation in observations:
        case = case_by_id[observation.sample_id]
        if observation.model_error:
            score_names = (
                "json_object_only",
                "schema_validity",
                "answer_exact",
                "answer_similarity",
                "answer_anls",
                "answer_token_f1",
                "numeric_match",
                "abstention_correct",
                "answer_correct",
                "evidence_page_precision",
                "evidence_page_recall",
                "evidence_page_f1",
                "evidence_coverage",
                "quote_answer_support",
                "quote_verifiability",
                "quote_grounding",
                "task_success",
            )
            scores = dict.fromkeys(score_names, 0.0)
            results.append(
                EvaluationResult(
                    sample_id=case.sample_id,
                    family_id=case.family_id,
                    status="inconclusive",
                    raw_output={"error": observation.model_error},
                    scores=scores,
                    reasons={name: observation.model_error for name in scores},
                    evidence_kind=observation.evidence_kind,
                )
            )
            continue

        page_texts = None
        if prepared_documents is not None:
            page_texts = text_cache.setdefault(
                case.document_id,
                load_page_texts(prepared_documents, case.document_id),
            )
        output, scores, reasons = score_output(
            observation.raw_output,
            case,
            page_texts=page_texts,
        )
        if output and any(
            evidence.page_number > observation.total_pages for evidence in output.evidence
        ):
            scores["evidence_coverage"] = 0.0
            scores["task_success"] = 0.0
            reasons["evidence_coverage"] = "근거 페이지가 PDF 범위를 벗어났습니다"
            reasons["task_success"] = "근거 페이지가 PDF 범위를 벗어남"
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
            )
        )
    return results
