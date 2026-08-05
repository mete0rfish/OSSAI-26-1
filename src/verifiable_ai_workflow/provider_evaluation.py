"""Week 2 오프라인 route 비교와 provider fault 리허설."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .comparison import (
    ComparisonContract,
    ComparisonReport,
    RouteDescriptor,
    compare_routes,
    sha256_file,
)
from .data.dataset import build_cases
from .evaluation.scoring import SCORING_PROFILE, score_observations
from .providers.recorded import RecordedProvider
from .schemas import EvaluationResult, ModelObservation


class Week2Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FaultEvent(Week2Model):
    status: Literal[
        "success",
        "auth_error",
        "rate_limit",
        "timeout",
        "server_error",
    ]
    provider: str
    actual_model: str | None = None
    latency_ms: float = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)


class FaultScenario(Week2Model):
    scenario_id: str
    mode: Literal["benchmark", "availability"]
    expected_actual_model: str
    events: list[FaultEvent] = Field(min_length=1)


class FaultOutcome(Week2Model):
    scenario_id: str
    final_status: Literal["success", "provider_error"]
    evidence_kind: Literal["test_only"]
    evaluation_mode: Literal["benchmark", "availability"]
    quality_eligible: bool
    comparison_eligible: bool
    retry_count: int
    attempt_count: int
    total_latency_ms: float
    estimated_cost_usd: float
    reason: str


class Week2Config(Week2Model):
    artifact_schema_version: Literal[2]
    baseline_route: RouteDescriptor
    candidate_route: RouteDescriptor
    candidate_overlay: str


def build_comparison_contract(project_root: str | Path) -> ComparisonContract:
    root = Path(project_root)
    return ComparisonContract(
        scoring_profile=SCORING_PROFILE,
        dataset_sha256=sha256_file(root / "data/cases/week-01-aihub.yaml"),
        prompt_sha256=sha256_file(root / "prompts/pdf-question-answer.md"),
        output_schema_sha256=sha256_file(root / "src/verifiable_ai_workflow/schemas/models.py"),
        scorer_sha256=sha256_file(root / "src/verifiable_ai_workflow/evaluation/scoring.py"),
        lockfile_sha256=sha256_file(root / "uv.lock"),
        temperature=0.0,
        max_output_tokens=500,
    )


def evaluate_recorded_route(
    project_root: str | Path,
    *,
    overlay_path: str | Path | None,
    requested_model: str,
    actual_model: str,
) -> list[EvaluationResult]:
    root = Path(project_root)
    cases = build_cases(root / "data/cases/week-01-aihub.yaml")
    provider = RecordedProvider(
        root / "data/recorded/week-02-gemma-baseline-responses.jsonl",
        overlay_path=overlay_path,
        requested_model=requested_model,
        actual_model=actual_model,
    )
    observations: list[ModelObservation] = []
    page_counts = {
        "MI2_240819_TY1_0012": 9,
        "MI2_240725_TY2_0002": 3,
    }
    for case in cases:
        response = provider.generate(case.sample_id, [])
        observations.append(
            ModelObservation(
                sample_id=case.sample_id,
                family_id=case.family_id,
                total_pages=page_counts[case.document_id],
                raw_output=response,
                model_call=provider.last_call,
                evidence_kind="test_only",
                evaluation_mode="benchmark",
            )
        )
    return score_observations(cases, observations)


def run_offline_comparison(project_root: str | Path) -> ComparisonReport:
    root = Path(project_root)
    config = Week2Config.model_validate(
        yaml.safe_load((root / "configs/week-02.yaml").read_text(encoding="utf-8"))
    )
    baseline = evaluate_recorded_route(
        root,
        overlay_path=None,
        requested_model=config.baseline_route.requested_model,
        actual_model=config.baseline_route.expected_actual_model,
    )
    candidate = evaluate_recorded_route(
        root,
        overlay_path=root / config.candidate_overlay,
        requested_model=config.candidate_route.requested_model,
        actual_model=config.candidate_route.expected_actual_model,
    )
    baseline_contract = build_comparison_contract(root)
    candidate_contract = build_comparison_contract(root)
    return compare_routes(
        baseline,
        candidate,
        baseline_route=config.baseline_route,
        candidate_route=config.candidate_route,
        baseline_contract=baseline_contract,
        candidate_contract=candidate_contract,
    )


def load_fault_scenarios(path: str | Path) -> list[FaultScenario]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [FaultScenario.model_validate(item) for item in payload["scenarios"]]


def rehearse_fault_scenario(scenario: FaultScenario) -> FaultOutcome:
    """저장 event를 실제 retry/fallback 상태 전이와 같은 규칙으로 해석한다."""

    final = scenario.events[-1]
    success = final.status == "success"
    actual_model_ok = (
        final.actual_model is not None and final.actual_model == scenario.expected_actual_model
    )
    benchmark = scenario.mode == "benchmark"
    quality_eligible = success and benchmark
    comparison_eligible = quality_eligible and actual_model_ok
    if not success:
        reason = "모든 제한 시도가 실패해 품질 분모에서 제외합니다."
    elif not benchmark:
        reason = "fallback 성공은 가용성 증거이며 고정 route 품질에서 제외합니다."
    elif final.actual_model is None:
        reason = "답은 보존하지만 actual model 미보고로 고정 route 비교를 중단합니다."
    elif not actual_model_ok:
        reason = "actual model drift로 고정 route 비교를 중단합니다."
    elif len(scenario.events) > 1:
        reason = "제한된 retry 뒤 성공했으며 모든 시도 비용과 시간을 보존합니다."
    else:
        reason = "첫 시도 성공입니다."
    return FaultOutcome(
        scenario_id=scenario.scenario_id,
        final_status="success" if success else "provider_error",
        evidence_kind="test_only",
        evaluation_mode=scenario.mode,
        quality_eligible=quality_eligible,
        comparison_eligible=comparison_eligible,
        retry_count=max(0, len(scenario.events) - 1),
        attempt_count=len(scenario.events),
        total_latency_ms=sum(event.latency_ms for event in scenario.events),
        estimated_cost_usd=sum(event.estimated_cost_usd for event in scenario.events),
        reason=reason,
    )


def rehearse_faults(path: str | Path) -> dict[str, Any]:
    outcomes = [rehearse_fault_scenario(scenario) for scenario in load_fault_scenarios(path)]
    return {
        "evidence_kind": "test_only",
        "scenario_count": len(outcomes),
        "final_status_counts": dict(Counter(outcome.final_status for outcome in outcomes)),
        "quality_eligible_count": sum(outcome.quality_eligible for outcome in outcomes),
        "comparison_eligible_count": sum(outcome.comparison_eligible for outcome in outcomes),
        "outcomes": [outcome.model_dump() for outcome in outcomes],
    }
