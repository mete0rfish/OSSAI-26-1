"""Week 2의 같은 문제를 같은 조건에서 두 route로 비교한다."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .schemas import EvaluationResult


class ComparisonModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComparisonContract(ComparisonModel):
    """모델 route 외에는 같아야 하는 비교 조건."""

    input_modality: Literal["page_images_only"] = "page_images_only"
    scoring_profile: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lockfile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temperature: float
    max_output_tokens: int = Field(gt=0)
    tool_policy: Literal["none"] = "none"

    @computed_field
    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(exclude={"sha256"}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class RouteDescriptor(ComparisonModel):
    logical_model: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    expected_actual_model: str = Field(min_length=1)


class CaseDiff(ComparisonModel):
    sample_id: str
    classification: Literal[
        "new_success",
        "new_failure",
        "unchanged",
        "not_comparable",
    ]
    baseline_status: str | None
    candidate_status: str | None
    baseline_task_success: bool | None
    candidate_task_success: bool | None
    latency_delta_ms: float | None = None
    input_tokens_delta: int | None = None
    output_tokens_delta: int | None = None


class RouteAggregate(ComparisonModel):
    record_count: int
    quality_eligible_count: int
    provider_error_count: int
    quality_coverage: float
    task_success_rate: float | None
    average_latency_ms: float | None


class ComparisonReport(ComparisonModel):
    baseline_route: RouteDescriptor
    candidate_route: RouteDescriptor
    baseline_contract_sha256: str
    candidate_contract_sha256: str
    baseline: RouteAggregate
    candidate: RouteAggregate
    case_diffs: list[CaseDiff]
    classification_counts: dict[str, int]
    task_success_delta_percentage_points: float | None
    invalid_comparison_reasons: list[str]
    evidence_kind: Literal["test_only", "live_quality"]
    automated_status: Literal["pass", "fail", "inconclusive"]
    automated_reason: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_success(result: EvaluationResult) -> bool | None:
    if result.provider_status not in {"success", "invalid_output"}:
        return None
    score = result.scores.get("task_success")
    return bool(score) if score is not None else None


def _quality_eligible(result: EvaluationResult) -> bool:
    return result.evaluation_mode == "benchmark" and result.provider_status in {
        "success",
        "invalid_output",
    }


def _aggregate(results: list[EvaluationResult]) -> RouteAggregate:
    eligible = [result for result in results if _quality_eligible(result)]
    task_scores = [
        float(result.scores["task_success"])
        for result in eligible
        if "task_success" in result.scores
    ]
    latencies = [
        float(result.model_call["latency_ms"])
        for result in results
        if result.model_call and result.model_call.get("latency_ms") is not None
    ]
    return RouteAggregate(
        record_count=len(results),
        quality_eligible_count=len(eligible),
        provider_error_count=sum(result.provider_status == "provider_error" for result in results),
        quality_coverage=len(eligible) / len(results) if results else 0.0,
        task_success_rate=(sum(task_scores) / len(task_scores) if task_scores else None),
        average_latency_ms=sum(latencies) / len(latencies) if latencies else None,
    )


def _metadata_delta(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    field: str,
) -> float | int | None:
    left = baseline.model_call.get(field) if baseline.model_call else None
    right = candidate.model_call.get(field) if candidate.model_call else None
    if left is None or right is None:
        return None
    return right - left


def _case_diff(
    sample_id: str,
    baseline: EvaluationResult | None,
    candidate: EvaluationResult | None,
) -> CaseDiff:
    left = _task_success(baseline) if baseline else None
    right = _task_success(candidate) if candidate else None
    if left is None or right is None:
        classification = "not_comparable"
    elif not left and right:
        classification = "new_success"
    elif left and not right:
        classification = "new_failure"
    else:
        classification = "unchanged"
    return CaseDiff(
        sample_id=sample_id,
        classification=classification,
        baseline_status=baseline.status if baseline else None,
        candidate_status=candidate.status if candidate else None,
        baseline_task_success=left,
        candidate_task_success=right,
        latency_delta_ms=(
            _metadata_delta(baseline, candidate, "latency_ms") if baseline and candidate else None
        ),
        input_tokens_delta=(
            _metadata_delta(baseline, candidate, "input_tokens") if baseline and candidate else None
        ),
        output_tokens_delta=(
            _metadata_delta(baseline, candidate, "output_tokens")
            if baseline and candidate
            else None
        ),
    )


def _model_metadata_problems(
    route: RouteDescriptor,
    results: list[EvaluationResult],
    label: str,
) -> list[str]:
    problems: list[str] = []
    for result in results:
        actual = result.model_call.get("actual_model") if result.model_call else None
        if not actual:
            problems.append(f"{label} actual model 미보고: {result.sample_id}")
        elif actual != route.expected_actual_model:
            problems.append(
                f"{label} actual model 불일치: {result.sample_id} "
                f"({actual} != {route.expected_actual_model})"
            )
    return problems


def compare_routes(
    baseline: list[EvaluationResult],
    candidate: list[EvaluationResult],
    *,
    baseline_route: RouteDescriptor,
    candidate_route: RouteDescriptor,
    baseline_contract: ComparisonContract,
    candidate_contract: ComparisonContract,
    max_regression_percentage_points: float = 0.0,
) -> ComparisonReport:
    """두 route를 비교하고 자동 상태와 문제별 변화를 함께 반환한다."""

    baseline_by_id = {result.sample_id: result for result in baseline}
    candidate_by_id = {result.sample_id: result for result in candidate}
    invalid: list[str] = []
    if len(baseline_by_id) != len(baseline):
        invalid.append("baseline sample_id 중복")
    if len(candidate_by_id) != len(candidate):
        invalid.append("candidate sample_id 중복")
    if baseline_contract.sha256 != candidate_contract.sha256:
        invalid.append("route 외 비교 조건 hash가 다릅니다")
    missing = sorted(baseline_by_id.keys() - candidate_by_id.keys())
    extra = sorted(candidate_by_id.keys() - baseline_by_id.keys())
    if missing:
        invalid.append(f"candidate 누락 sample: {missing}")
    if extra:
        invalid.append(f"candidate 추가 sample: {extra}")
    invalid.extend(_model_metadata_problems(baseline_route, baseline, "baseline"))
    invalid.extend(_model_metadata_problems(candidate_route, candidate, "candidate"))
    all_ids = sorted(baseline_by_id.keys() | candidate_by_id.keys())
    diffs = [
        _case_diff(
            sample_id,
            baseline_by_id.get(sample_id),
            candidate_by_id.get(sample_id),
        )
        for sample_id in all_ids
    ]
    counts = dict(Counter(diff.classification for diff in diffs))
    baseline_aggregate = _aggregate(baseline)
    candidate_aggregate = _aggregate(candidate)
    if (
        baseline_aggregate.task_success_rate is None
        or candidate_aggregate.task_success_rate is None
    ):
        delta = None
    else:
        delta = (candidate_aggregate.task_success_rate - baseline_aggregate.task_success_rate) * 100

    evidence_values = {result.evidence_kind for result in [*baseline, *candidate]}
    live_evidence = evidence_values == {"live_quality"}
    if invalid:
        status = "inconclusive"
        reason = "비교 조건 또는 model metadata가 달라 자동 판정할 수 없습니다."
    elif not live_evidence:
        status = "inconclusive"
        reason = "저장 응답은 코드 리허설일 뿐 실제 route 품질 증거가 아닙니다."
    elif counts.get("new_failure", 0) > 0 or (
        delta is not None and delta < -max_regression_percentage_points
    ):
        status = "fail"
        reason = "새 실패 또는 허용 범위를 넘는 품질 하락이 있습니다."
    else:
        status = "pass"
        reason = "동일한 비교 조건에서 새 실패와 허용 범위 밖 하락이 없습니다."
    return ComparisonReport(
        baseline_route=baseline_route,
        candidate_route=candidate_route,
        baseline_contract_sha256=baseline_contract.sha256,
        candidate_contract_sha256=candidate_contract.sha256,
        baseline=baseline_aggregate,
        candidate=candidate_aggregate,
        case_diffs=diffs,
        classification_counts=counts,
        task_success_delta_percentage_points=delta,
        invalid_comparison_reasons=invalid,
        evidence_kind="live_quality" if live_evidence else "test_only",
        automated_status=status,
        automated_reason=reason,
    )
