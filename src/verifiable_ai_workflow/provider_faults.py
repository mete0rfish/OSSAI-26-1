"""저장된 provider 오류·retry·fallback 상황을 품질 결과와 분리한다."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FaultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FaultEvent(FaultModel):
    status: Literal["success", "auth_error", "rate_limit", "timeout", "server_error"]
    provider: str
    actual_model: str | None = None
    latency_ms: float = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)


class FaultScenario(FaultModel):
    scenario_id: str
    mode: Literal["benchmark", "availability"]
    expected_actual_model: str
    events: list[FaultEvent] = Field(min_length=1)


class FaultOutcome(FaultModel):
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


def load_fault_scenarios(path: str | Path) -> list[FaultScenario]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [FaultScenario.model_validate(item) for item in payload["scenarios"]]


def rehearse_fault_scenario(scenario: FaultScenario) -> FaultOutcome:
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
