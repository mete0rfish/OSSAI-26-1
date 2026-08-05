from pathlib import Path

from verifiable_ai_workflow.comparison import compare_routes
from verifiable_ai_workflow.provider_evaluation import (
    build_comparison_contract,
    evaluate_recorded_route,
    run_offline_comparison,
)


def test_offline_route_comparison_exposes_case_changes(project_root: Path) -> None:
    report = run_offline_comparison(project_root)

    assert report.baseline.record_count == 40
    assert report.candidate.record_count == 40
    assert report.classification_counts == {
        "new_success": 2,
        "unchanged": 38,
    }
    assert report.evidence_kind == "test_only"
    assert report.automated_status == "inconclusive"


def test_contract_change_makes_comparison_inconclusive(project_root: Path) -> None:
    report = run_offline_comparison(project_root)
    baseline = evaluate_recorded_route(
        project_root,
        overlay_path=None,
        requested_model=report.baseline_route.requested_model,
        actual_model=report.baseline_route.expected_actual_model,
    )
    candidate = evaluate_recorded_route(
        project_root,
        overlay_path=project_root / "data/scenarios/week-02-route-b-overrides.jsonl",
        requested_model=report.candidate_route.requested_model,
        actual_model=report.candidate_route.expected_actual_model,
    )
    contract = build_comparison_contract(project_root)
    changed = contract.model_copy(update={"max_output_tokens": contract.max_output_tokens + 1})

    invalid = compare_routes(
        baseline,
        candidate,
        baseline_route=report.baseline_route,
        candidate_route=report.candidate_route,
        baseline_contract=contract,
        candidate_contract=changed,
    )

    assert invalid.automated_status == "inconclusive"
    assert invalid.invalid_comparison_reasons == ["route 외 비교 조건 hash가 다릅니다"]


def test_actual_model_missing_stops_fixed_route_comparison(
    project_root: Path,
) -> None:
    report = run_offline_comparison(project_root)
    baseline = evaluate_recorded_route(
        project_root,
        overlay_path=None,
        requested_model=report.baseline_route.requested_model,
        actual_model=report.baseline_route.expected_actual_model,
    )
    candidate = evaluate_recorded_route(
        project_root,
        overlay_path=project_root / "data/scenarios/week-02-route-b-overrides.jsonl",
        requested_model=report.candidate_route.requested_model,
        actual_model=report.candidate_route.expected_actual_model,
    )
    first = candidate[0]
    candidate[0] = first.model_copy(
        update={"model_call": {**first.model_call, "actual_model": None}}
    )
    contract = build_comparison_contract(project_root)

    invalid = compare_routes(
        baseline,
        candidate,
        baseline_route=report.baseline_route,
        candidate_route=report.candidate_route,
        baseline_contract=contract,
        candidate_contract=contract,
    )

    assert invalid.automated_status == "inconclusive"
    assert any("actual model 미보고" in reason for reason in invalid.invalid_comparison_reasons)
