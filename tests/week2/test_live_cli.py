import importlib.util
from pathlib import Path
from types import SimpleNamespace

from verifiable_ai_workflow.live_provider_comparison import load_week2_live_config


def _execution(*, invalid_outputs: int = 0, model_mismatches: int = 0):
    return SimpleNamespace(
        summary=SimpleNamespace(
            baseline_observation_count=1,
            candidate_observation_count=1,
            baseline_provider_errors=0,
            candidate_provider_errors=0,
            baseline_invalid_outputs=invalid_outputs,
            candidate_invalid_outputs=0,
        ),
        baseline_provenance=SimpleNamespace(
            actual_model_mismatch_count=model_mismatches,
        ),
        candidate_provenance=SimpleNamespace(
            actual_model_mismatch_count=0,
        ),
    )


def test_probe_succeeds_only_with_two_valid_expected_model_responses(
    project_root: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "compare_live_provider_routes_test",
        project_root / "scripts/compare_live_provider_routes.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._probe_succeeded(_execution())
    assert not module._probe_succeeded(_execution(invalid_outputs=1))
    assert not module._probe_succeeded(_execution(model_mismatches=1))
    assert module.PROBE_SAMPLE_IDS == (
        "aihub-report-r01",
        "aihub-report-r03",
        "aihub-report-r31",
    )


def test_live_comparison_uses_model_neutral_improved_prompt(project_root: Path) -> None:
    config = load_week2_live_config(project_root / "configs/week-02-live.yaml")
    prompt = (project_root / config.paths.prompt).read_text(encoding="utf-8")

    assert config.paths.prompt == "prompts/pdf-question-answer-json-only.md"
    assert "/no_think" not in prompt
    assert "값과 단위만" in prompt
    assert "두 번째 JSON을 절대 출력하지 않습니다" in prompt


def test_probe_and_full_run_use_different_git_clean_rules(project_root: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "compare_live_provider_routes_git_rule_test",
        project_root / "scripts/compare_live_provider_routes.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._require_clean_git("aihub-report-r01") is False
    assert module._require_clean_git(None) is True
