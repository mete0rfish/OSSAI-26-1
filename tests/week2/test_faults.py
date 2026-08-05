from pathlib import Path

from verifiable_ai_workflow.provider_evaluation import (
    load_fault_scenarios,
    rehearse_fault_scenario,
    rehearse_faults,
)


def test_six_fault_scenarios_keep_quality_and_availability_separate(
    project_root: Path,
) -> None:
    path = project_root / "data/scenarios/week-02-provider-faults.yaml"
    report = rehearse_faults(path)

    assert report["scenario_count"] == 6
    assert report["final_status_counts"] == {
        "provider_error": 2,
        "success": 4,
    }
    assert report["quality_eligible_count"] == 3
    assert report["comparison_eligible_count"] == 1


def test_fallback_success_is_not_benchmark_quality(project_root: Path) -> None:
    scenarios = load_fault_scenarios(project_root / "data/scenarios/week-02-provider-faults.yaml")
    fallback = next(
        scenario for scenario in scenarios if scenario.scenario_id == "fallback-success"
    )

    outcome = rehearse_fault_scenario(fallback)

    assert outcome.final_status == "success"
    assert outcome.evaluation_mode == "availability"
    assert not outcome.quality_eligible
    assert outcome.attempt_count == 2
    assert outcome.estimated_cost_usd == 0.004


def test_retry_cost_and_latency_include_failed_attempt(project_root: Path) -> None:
    scenarios = load_fault_scenarios(project_root / "data/scenarios/week-02-provider-faults.yaml")
    retry = next(
        scenario for scenario in scenarios if scenario.scenario_id == "rate-limit-then-success"
    )

    outcome = rehearse_fault_scenario(retry)

    assert outcome.retry_count == 1
    assert outcome.total_latency_ms == 250
    assert outcome.estimated_cost_usd == 0.002
    assert outcome.quality_eligible
    assert outcome.comparison_eligible
