"""Gemma prompt A/B의 통제 조건과 사례별 변화를 비교한다."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .comparison import sha256_file
from .data.dataset import build_cases
from .evaluation.scoring import SCORING_PROFILE, score_observations
from .schemas import EvaluationResult, ModelObservation

CONTROLLED_PROVENANCE_FIELDS = (
    "git_sha",
    "dataset_sha256",
    "input_manifest_content_sha256",
    "lockfile_sha256",
    "schema_sha256",
    "scorer_sha256",
    "workflow_sha256",
)
METRIC_NAMES = (
    "task_success",
    "answer_correct",
    "schema_validity",
    "json_object_only",
    "numeric_match",
    "evidence_page_f1",
    "quote_answer_support",
)


class PromptComparisonModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricDelta(PromptComparisonModel):
    baseline: float = Field(ge=0.0, le=1.0)
    candidate: float = Field(ge=0.0, le=1.0)
    delta_percentage_points: float


class PromptCaseDiff(PromptComparisonModel):
    sample_id: str
    classification: Literal["new_success", "new_failure", "unchanged", "not_comparable"]
    baseline_task_success: bool | None
    candidate_task_success: bool | None
    baseline_answer: str | None
    candidate_answer: str | None


class PromptComparisonReport(PromptComparisonModel):
    artifact_schema_version: Literal[2] = 2
    experiment_type: Literal["prompt_only"] = "prompt_only"
    score_source: Literal["stored_results", "rescored_observations"]
    scoring_profile: str
    effective_dataset_sha256: str | None
    effective_scorer_sha256: str | None
    baseline_run_id: str
    candidate_run_id: str
    baseline_prompt_sha256: str | None
    candidate_prompt_sha256: str | None
    controlled_provenance: dict[str, str]
    quality_eligible_counts: dict[str, int]
    provider_error_counts: dict[str, int]
    metric_deltas: dict[str, MetricDelta]
    classification_counts: dict[str, int]
    new_success_ids: list[str]
    new_failure_ids: list[str]
    not_comparable_ids: list[str]
    case_diffs: list[PromptCaseDiff]
    invalid_reasons: list[str]
    evidence_kind: Literal["live_quality"] = "live_quality"
    automated_status: Literal["pass", "fail", "inconclusive"]
    automated_reason: str


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object가 필요합니다: {path}")
    return value


def _load_results(path: Path) -> list[EvaluationResult]:
    return [
        EvaluationResult.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_run(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], list[EvaluationResult]]:
    root = Path(run_dir)
    summary_path = root / "summary.json"
    manifest_path = root / "run-manifest.json"
    results_path = root / "results.jsonl"
    if not summary_path.is_file() or not manifest_path.is_file() or not results_path.is_file():
        raise ValueError(
            f"summary.json, run-manifest.json과 results.jsonl이 모두 필요합니다: {root}"
        )
    return _load_json(summary_path), _load_json(manifest_path), _load_results(results_path)


def _rescore_run(run_dir: str | Path, case_authoring_path: str | Path) -> list[EvaluationResult]:
    observations_path = Path(run_dir) / "observations.jsonl"
    if not observations_path.is_file():
        raise ValueError(f"현재 채점기로 다시 계산할 observations.jsonl이 없습니다: {run_dir}")
    observations = [
        ModelObservation.model_validate_json(line)
        for line in observations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return score_observations(build_cases(case_authoring_path), observations)


def _run_problems(
    label: str,
    summary: dict[str, Any],
    results: list[EvaluationResult],
) -> list[str]:
    problems: list[str] = []
    expected_checks = {
        "status": "complete",
        "observed_status": "complete",
        "probe_only": False,
        "evidence_kind": "live_quality",
        "evaluation_mode": "benchmark",
        "fallback_enabled": False,
        "replay_enabled": False,
        "live_call_performed": True,
        "provider_error_count": 0,
        "model_drift_count": 0,
    }
    for field, expected in expected_checks.items():
        if summary.get(field) != expected:
            problems.append(f"{label} {field}={summary.get(field)!r}, expected={expected!r}")

    record_count = summary.get("record_count")
    target_count = summary.get("target_count")
    if record_count != target_count or record_count != len(results):
        problems.append(
            f"{label} 결과 수 불일치: record={record_count}, "
            f"target={target_count}, results={len(results)}"
        )
    if target_count != 40:
        problems.append(f"{label} Week 1 전체 40건 실행이 아닙니다: {target_count}")

    sample_ids = [result.sample_id for result in results]
    if len(sample_ids) != len(set(sample_ids)):
        problems.append(f"{label} sample_id 중복")
    if any(result.evidence_kind != "live_quality" for result in results):
        problems.append(f"{label}에 live_quality가 아닌 결과가 있습니다")
    if any(result.provider_status == "provider_error" for result in results):
        problems.append(f"{label}에 provider 오류 결과가 있습니다")

    expected_model = summary.get("expected_actual_model")
    for result in results:
        actual_model = result.model_call.get("actual_model") if result.model_call else None
        if actual_model != expected_model:
            problems.append(
                f"{label} actual model 불일치: {result.sample_id} "
                f"({actual_model!r} != {expected_model!r})"
            )
    return problems


def _quality_eligible(result: EvaluationResult) -> bool:
    return result.provider_status in {"success", "invalid_output"}


def _metric_average(results: list[EvaluationResult], metric: str) -> float:
    eligible = [result for result in results if _quality_eligible(result)]
    values = [result.scores[metric] for result in eligible if metric in result.scores]
    if len(values) != len(eligible) or not values:
        raise ValueError(f"품질 판정 가능한 모든 결과에 {metric} 점수가 필요합니다")
    return sum(values) / len(values)


def compare_prompt_runs(
    baseline_run_dir: str | Path,
    candidate_run_dir: str | Path,
    *,
    case_authoring_path: str | Path | None = None,
) -> PromptComparisonReport:
    """같은 model·입력·코드에서 prompt만 다른 두 full live run을 비교한다."""

    baseline_summary, baseline_manifest, baseline_results = _load_run(baseline_run_dir)
    candidate_summary, candidate_manifest, candidate_results = _load_run(candidate_run_dir)
    if case_authoring_path is None:
        score_source: Literal["stored_results", "rescored_observations"] = "stored_results"
        effective_dataset_sha256 = None
        effective_scorer_sha256 = None
    else:
        score_source = "rescored_observations"
        effective_dataset_sha256 = sha256_file(case_authoring_path)
        effective_scorer_sha256 = sha256_file(Path(__file__).parent / "evaluation/scoring.py")
        baseline_results = _rescore_run(baseline_run_dir, case_authoring_path)
        candidate_results = _rescore_run(candidate_run_dir, case_authoring_path)
    invalid = [
        *_run_problems("baseline", baseline_summary, baseline_results),
        *_run_problems("candidate", candidate_summary, candidate_results),
    ]

    if baseline_summary.get("requested_model") != candidate_summary.get("requested_model"):
        invalid.append("요청 model이 다릅니다")
    if baseline_summary.get("expected_actual_model") != candidate_summary.get(
        "expected_actual_model"
    ):
        invalid.append("기대 actual model이 다릅니다")
    if baseline_summary.get("actual_models") != candidate_summary.get("actual_models"):
        invalid.append("관찰한 actual model 목록이 다릅니다")

    baseline_contract = baseline_manifest.get("contract") or {}
    candidate_contract = candidate_manifest.get("contract") or {}
    for field in (
        "provider",
        "evaluation_mode",
        "evidence_kind",
        "fallback_enabled",
        "replay_enabled",
        "caps",
    ):
        if baseline_contract.get(field) != candidate_contract.get(field):
            invalid.append(f"prompt 외 실행 조건이 다릅니다: {field}")
    if baseline_manifest.get("input_manifest_sha256") != candidate_manifest.get(
        "input_manifest_sha256"
    ):
        invalid.append("prompt 외 실행 조건이 다릅니다: input_manifest_sha256")
    for label, summary, manifest in (
        ("baseline", baseline_summary, baseline_manifest),
        ("candidate", candidate_summary, candidate_manifest),
    ):
        if manifest.get("status") != "complete":
            invalid.append(f"{label} run manifest가 complete가 아닙니다")
        if manifest.get("contract", {}).get("run_id") != summary.get("run_id"):
            invalid.append(f"{label} summary와 run manifest의 run_id가 다릅니다")

    baseline_provenance = baseline_summary.get("provenance") or {}
    candidate_provenance = candidate_summary.get("provenance") or {}
    controlled: dict[str, str] = {}
    for field in CONTROLLED_PROVENANCE_FIELDS:
        baseline_value = baseline_provenance.get(field)
        candidate_value = candidate_provenance.get(field)
        if not isinstance(baseline_value, str) or not isinstance(candidate_value, str):
            invalid.append(f"통제 provenance 누락: {field}")
        elif baseline_value != candidate_value:
            invalid.append(f"prompt 외 통제값 불일치: {field}")
        else:
            controlled[field] = baseline_value

    baseline_prompt = baseline_provenance.get("prompt_sha256")
    candidate_prompt = candidate_provenance.get("prompt_sha256")
    if not isinstance(baseline_prompt, str) or not isinstance(candidate_prompt, str):
        invalid.append("prompt hash가 없습니다")
    elif baseline_prompt == candidate_prompt:
        invalid.append("prompt 후보가 기준 prompt와 같습니다")

    baseline_by_id = {result.sample_id: result for result in baseline_results}
    candidate_by_id = {result.sample_id: result for result in candidate_results}
    if baseline_by_id.keys() != candidate_by_id.keys():
        missing = sorted(baseline_by_id.keys() - candidate_by_id.keys())
        extra = sorted(candidate_by_id.keys() - baseline_by_id.keys())
        invalid.append(f"sample 집합 불일치: missing={missing}, extra={extra}")

    case_diffs: list[PromptCaseDiff] = []
    for sample_id in sorted(baseline_by_id.keys() & candidate_by_id.keys()):
        baseline = baseline_by_id[sample_id]
        candidate = candidate_by_id[sample_id]
        baseline_comparable = _quality_eligible(baseline) and (
            baseline.model_call.get("actual_model") == baseline_summary.get("expected_actual_model")
        )
        candidate_comparable = _quality_eligible(candidate) and (
            candidate.model_call.get("actual_model")
            == candidate_summary.get("expected_actual_model")
        )
        baseline_success = (
            baseline.scores.get("task_success") == 1.0 if baseline_comparable else None
        )
        candidate_success = (
            candidate.scores.get("task_success") == 1.0 if candidate_comparable else None
        )
        if baseline_success is None or candidate_success is None:
            classification = "not_comparable"
        elif not baseline_success and candidate_success:
            classification = "new_success"
        elif baseline_success and not candidate_success:
            classification = "new_failure"
        else:
            classification = "unchanged"
        case_diffs.append(
            PromptCaseDiff(
                sample_id=sample_id,
                classification=classification,
                baseline_task_success=baseline_success,
                candidate_task_success=candidate_success,
                baseline_answer=baseline.output.answer if baseline.output else None,
                candidate_answer=candidate.output.answer if candidate.output else None,
            )
        )

    counts = dict(Counter(item.classification for item in case_diffs))
    metrics: dict[str, MetricDelta] = {}
    for metric in METRIC_NAMES:
        baseline_average = _metric_average(baseline_results, metric)
        candidate_average = _metric_average(candidate_results, metric)
        metrics[metric] = MetricDelta(
            baseline=round(baseline_average, 4),
            candidate=round(candidate_average, 4),
            delta_percentage_points=round(
                (candidate_average - baseline_average) * 100,
                2,
            ),
        )

    new_success_ids = [
        item.sample_id for item in case_diffs if item.classification == "new_success"
    ]
    new_failure_ids = [
        item.sample_id for item in case_diffs if item.classification == "new_failure"
    ]
    not_comparable_ids = [
        item.sample_id for item in case_diffs if item.classification == "not_comparable"
    ]
    if invalid:
        status: Literal["pass", "fail", "inconclusive"] = "inconclusive"
        reason = "prompt 외 조건이 달라 성능 변화를 prompt 효과로 해석할 수 없습니다."
    elif metrics["task_success"].delta_percentage_points <= 0:
        status = "fail"
        reason = "Gemma 후보 prompt가 전체 성공률을 높이지 못했습니다."
    elif new_failure_ids:
        status = "fail"
        reason = "전체 성공률은 올랐지만 새 실패가 있어 사례 검토가 필요합니다."
    else:
        status = "pass"
        reason = "같은 조건에서 전체 성공률이 올랐고 새 실패가 없습니다."

    return PromptComparisonReport(
        score_source=score_source,
        scoring_profile=SCORING_PROFILE,
        effective_dataset_sha256=effective_dataset_sha256,
        effective_scorer_sha256=effective_scorer_sha256,
        baseline_run_id=str(baseline_summary.get("run_id", "")),
        candidate_run_id=str(candidate_summary.get("run_id", "")),
        baseline_prompt_sha256=(baseline_prompt if isinstance(baseline_prompt, str) else None),
        candidate_prompt_sha256=(candidate_prompt if isinstance(candidate_prompt, str) else None),
        controlled_provenance=controlled,
        quality_eligible_counts={
            "baseline": sum(_quality_eligible(result) for result in baseline_results),
            "candidate": sum(_quality_eligible(result) for result in candidate_results),
        },
        provider_error_counts={
            "baseline": sum(
                result.provider_status == "provider_error" for result in baseline_results
            ),
            "candidate": sum(
                result.provider_status == "provider_error" for result in candidate_results
            ),
        },
        metric_deltas=metrics,
        classification_counts=counts,
        new_success_ids=new_success_ids,
        new_failure_ids=new_failure_ids,
        not_comparable_ids=not_comparable_ids,
        case_diffs=case_diffs,
        invalid_reasons=invalid,
        automated_status=status,
        automated_reason=reason,
    )
