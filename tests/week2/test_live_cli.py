import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from verifiable_ai_workflow.comparison import ComparisonContract
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


def test_default_output_dir_uses_mode_and_utc_timestamp(project_root: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "compare_live_provider_routes_output_test",
        project_root / "scripts/compare_live_provider_routes.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now = datetime(2026, 8, 8, 12, 34, 56, 123456, tzinfo=UTC)

    assert module._default_output_dir("aihub-report-r01", now=now) == (
        project_root
        / "reports/week-02-live/probe-aihub-report-r01-20260808T123456123456Z"
    )
    assert module._default_output_dir(None, now=now) == (
        project_root / "reports/week-02-live/full-20260808T123456123456Z"
    )


def test_provider_rescore_keeps_source_and_effective_contracts_separate(
    project_root: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "rescore_provider_comparison_test",
        project_root / "scripts/rescore_provider_comparison.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_contract = "a" * 64
    effective = ComparisonContract(
        scoring_profile="current",
        dataset_sha256="b" * 64,
        prompt_sha256="c" * 64,
        output_schema_sha256="d" * 64,
        scorer_sha256="e" * 64,
        lockfile_sha256="f" * 64,
        max_output_tokens=500,
    )

    source_manifest = {
        "scoring_profile": "original",
        "case_authoring_sha256": "1" * 64,
        "prompt_sha256": "2" * 64,
        "output_schema_sha256": "3" * 64,
        "scorer_sha256": "4" * 64,
    }
    effective = module._keep_source_prompt(effective, source_manifest)
    context = module._provenance_context(
        source_manifest,
        {
            "run_id": "source-run",
            "input_manifest_sha256": "5" * 64,
            "comparison_contract_sha256": source_contract,
        },
        effective,
    )

    assert context["source_execution"]["comparison_contract_sha256"] == source_contract
    assert context["effective_rescoring"]["sha256"] == effective.sha256
    assert context["effective_rescoring"]["sha256"] != source_contract
    assert context["source_execution"]["prompt_sha256"] == "2" * 64
    assert context["effective_rescoring"]["prompt_sha256"] == "2" * 64
    assert context["source_execution"]["dataset_sha256"] != (
        context["effective_rescoring"]["dataset_sha256"]
    )
    assert context["source_execution"]["output_schema_sha256"] != (
        context["effective_rescoring"]["output_schema_sha256"]
    )
    assert context["source_execution"]["scorer_sha256"] != (
        context["effective_rescoring"]["scorer_sha256"]
    )
    assert context["source_execution"]["lockfile_sha256"] is None
