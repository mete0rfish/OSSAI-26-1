from pathlib import Path

import yaml


def load_manifest(project_root: Path) -> dict:
    return yaml.safe_load(
        (project_root / "configs/live-egress.yaml").read_text(encoding="utf-8")
    )


def test_live_egress_manifest_is_scoped_to_week_01_and_week_02(
    project_root: Path,
) -> None:
    payload = load_manifest(project_root)

    assert payload["status"] == "week_02_live_completed_hold"
    assert payload["week_02_live_results"] == (
        "reports/week-02-live/full-improved-20260805-rerun-01/summary.json"
    )
    assert set(payload["destinations"]) == {"nvidia_nim", "google_ai_studio"}
    assert set(payload["runs"]) == {"week_01_task", "week_02_two_provider"}
    approval = payload["approval_conditions"]
    assert approval["scope"] == "week_01_to_week_02_only"
    assert approval["sealed_test_allowed"] is False
    assert approval["current_decision"] == "HOLD"


def test_week_02_actual_execution_matches_recorded_result(project_root: Path) -> None:
    actual = load_manifest(project_root)["runs"]["week_02_two_provider"][
        "actual_execution"
    ]

    assert actual["full_requests"] == 80
    assert actual["full_successful_responses"] == 80
    assert actual["baseline_provider_errors"] == 0
    assert actual["candidate_provider_errors"] == 0
    assert actual["baseline_task_successes"] == 27
    assert actual["candidate_task_successes"] == 35
    assert actual["new_success"] == 8
    assert actual["new_failure"] == 0
    assert actual["automated_status"] == "pass"
    assert actual["release_claim"] is False
    assert actual["human_decision"] == "HOLD"


def test_every_external_run_fixes_endpoint_key_model_and_caps(
    project_root: Path,
) -> None:
    payload = load_manifest(project_root)

    for run in payload["runs"].values():
        contracts = run.get("provider_contracts") or {"only": run["provider_contract"]}
        for contract in contracts.values():
            assert contract["endpoint"] in {
                "https://integrate.api.nvidia.com/v1",
                "https://generativelanguage.googleapis.com/v1beta",
            }
            assert contract["api_key_env"] in {
                "NVIDIA_NIM_API_KEY",
                "GEMINI_API_KEY",
            }
            assert any("model" in key for key in contract)
        for mode in ("probe_caps", "full_caps"):
            caps = run["planned_scope"][mode]
            assert caps["requests"] > 0
            assert caps["input_tokens"] > 0
            assert caps["output_tokens"] > 0
            assert caps["retries"] == 0
            assert caps["cost_usd"] > 0
            assert caps["wall_seconds"] > 0

    week2 = yaml.safe_load(
        (project_root / "configs/week-02-live.yaml").read_text(encoding="utf-8")
    )
    assert week2["candidate_route"]["model"] == "gemini/gemini-3.5-flash-lite"
    assert week2["candidate_route"]["api_key_env"] == "GEMINI_API_KEY"
    assert week2["candidate_route"]["api_base"] == (
        "https://generativelanguage.googleapis.com/v1beta"
    )


def test_live_egress_manifest_contains_no_secret_value(project_root: Path) -> None:
    text = (project_root / "configs/live-egress.yaml").read_text(encoding="utf-8")

    assert "nvapi-" not in text
    assert "AIza" not in text
    assert "API_KEY=" not in text
