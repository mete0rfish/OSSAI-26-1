from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.live_execution import LiveExecutionError
from verifiable_ai_workflow.schemas import EvaluationResult, ModelObservation


@pytest.fixture
def live_runner(project_root: Path) -> ModuleType:
    script = project_root / "scripts/run_nvidia_nim.py"
    spec = importlib.util.spec_from_file_location("week01_live_runner_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_run_level_status_is_always_inconclusive(live_runner: ModuleType) -> None:
    status, observed_status = live_runner._classify_run_status(
        probe_only=True,
        blocked=False,
        complete=True,
        provider_error_count=0,
        model_drift_count=0,
        failed_result_count=0,
    )

    assert status == "inconclusive"
    assert observed_status == "complete"


def test_full_run_separates_execution_completion_from_quality(live_runner: ModuleType) -> None:
    status, observed_status = live_runner._classify_run_status(
        probe_only=False,
        blocked=False,
        complete=True,
        provider_error_count=0,
        model_drift_count=0,
        failed_result_count=0,
    )

    assert status == "pass"
    assert observed_status == "complete"

    status, observed_status = live_runner._classify_run_status(
        probe_only=False,
        blocked=False,
        complete=True,
        provider_error_count=0,
        model_drift_count=0,
        failed_result_count=1,
    )

    assert status == "fail"
    assert observed_status == "complete"


def test_invalid_full_run_is_inconclusive(live_runner: ModuleType) -> None:
    status, observed_status = live_runner._classify_run_status(
        probe_only=False,
        blocked=False,
        complete=True,
        provider_error_count=1,
        model_drift_count=0,
        failed_result_count=0,
    )

    assert status == "inconclusive"
    assert observed_status == "inconclusive"


def test_model_drift_and_provider_error_are_counted_separately(
    live_runner: ModuleType,
) -> None:
    drift = ModelObservation(
        sample_id="sample-1",
        family_id="family-1",
        total_pages=1,
        model_error="RuntimeError: actual model mismatch",
        model_call={
            "actual_model": "other/model",
            "actual_model_matches_expected": False,
            "error_type": "ActualModelMismatch",
        },
        evidence_kind="live_quality",
    )
    error = drift.model_copy(
        update={"model_call": {"error_type": "TimeoutError", "actual_model": None}}
    )

    assert live_runner._is_model_drift(drift)
    assert not live_runner._is_model_drift(error)


def test_provider_error_is_excluded_from_quality_average(live_runner: ModuleType) -> None:
    scores = {"task_success": 1.0, "answer_correct": 1.0}
    success = EvaluationResult(
        sample_id="sample-1",
        family_id="family-1",
        status="passed",
        scores=scores,
        reasons={},
        evidence_kind="live_quality",
    )
    error = success.model_copy(
        update={
            "sample_id": "sample-2",
            "status": "inconclusive",
            "provider_status": "provider_error",
            "scores": dict.fromkeys(scores, 0.0),
        }
    )

    count, averages = live_runner._quality_score_averages([success, error])

    assert count == 1
    assert averages["task_success"] == 1.0


def test_local_case_copy_must_exactly_match_tracked_non_sealed_40(
    live_runner: ModuleType,
    project_root: Path,
) -> None:
    canonical = build_cases(project_root / "data/cases/week-01-aihub.yaml")

    approved = live_runner._require_approved_case_copy(
        canonical_cases=canonical,
        local_cases=list(canonical),
    )

    assert approved == canonical
    changed = list(canonical)
    changed[0] = changed[0].model_copy(update={"question": "승인되지 않은 질문"})
    with pytest.raises(ValueError, match="exact 40건"):
        live_runner._require_approved_case_copy(
            canonical_cases=canonical,
            local_cases=changed,
        )

def test_live_runner_allows_only_reviewed_nvidia_configs(
    live_runner: ModuleType,
    project_root: Path,
) -> None:
    assert live_runner._require_approved_config("configs/nvidia-nim.yaml") == (
        project_root / "configs/nvidia-nim.yaml"
    )
    assert live_runner._require_approved_config("configs/nvidia-nim-gemma4.yaml") == (
        project_root / "configs/nvidia-nim-gemma4.yaml"
    )
    assert live_runner._require_approved_config(
        "configs/nvidia-nim-gemma4-baseline.yaml"
    ) == (project_root / "configs/nvidia-nim-gemma4-baseline.yaml")

    with pytest.raises(LiveExecutionError, match="승인된 NVIDIA NIM 설정"):
        live_runner._require_approved_config("configs/week-01.yaml")


def test_only_full_quality_run_requires_clean_git(
    live_runner: ModuleType,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = live_runner.load_settings(project_root / "configs/nvidia-nim.yaml")
    case = build_cases(project_root / "data/cases/week-01-aihub.yaml")[0]
    monkeypatch.setattr(live_runner, "_git_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(live_runner, "_sha256_file", lambda _path: "b" * 64)

    exploratory = live_runner._build_provenance(
        settings=settings,
        config_path=project_root / "configs/nvidia-nim.yaml",
        cases=[case],
        input_manifest={"sample_ids": [case.sample_id]},
        catalog_verified_on=live_runner.date.today(),
        pricing_verified_on=live_runner.date.today(),
        require_clean_git=False,
    )
    assert exploratory["git_clean"] is False
    assert exploratory["config_path"] == "configs/nvidia-nim.yaml"
    assert exploratory["pricing_verified_on"] == live_runner.date.today().isoformat()

    with pytest.raises(RuntimeError, match="전체 품질 실행"):
        live_runner._build_provenance(
            settings=settings,
            config_path=project_root / "configs/nvidia-nim.yaml",
            cases=[case],
            input_manifest={"sample_ids": [case.sample_id]},
            catalog_verified_on=live_runner.date.today(),
            pricing_verified_on=live_runner.date.today(),
            require_clean_git=True,
        )


def test_local_prompt_must_be_an_existing_local_data_file(
    live_runner: ModuleType,
    project_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = live_runner.load_settings(project_root / "configs/nvidia-nim-gemma4-baseline.yaml")
    local_data = tmp_path / "local-data"
    local_data.mkdir()
    prompt = local_data / "my-prompt.md"
    prompt.write_text("JSON 하나만 반환합니다.\n", encoding="utf-8")
    monkeypatch.setattr(live_runner, "PROJECT_ROOT", tmp_path)

    changed = live_runner._with_local_prompt(settings, prompt)
    assert changed.paths.prompt == "local-data/my-prompt.md"

    outside = tmp_path / "outside.md"
    outside.write_text("허용하지 않는 위치\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local-data"):
        live_runner._with_local_prompt(settings, outside)


def test_week2_baseline_must_be_complete_clean_same_release(
    live_runner: ModuleType,
    project_root: Path,
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    baseline_prompt_hash = live_runner._sha256_file(
        project_root / "prompts/pdf-question-answer.md"
    )
    (baseline / "prompt.md").write_bytes(
        (project_root / "prompts/pdf-question-answer.md").read_bytes()
    )
    provenance = {
        "git_clean": True,
        "git_sha": "a" * 40,
        "config_path": "configs/nvidia-nim-gemma4-baseline.yaml",
        "config_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "input_manifest_content_sha256": "d" * 64,
        "lockfile_sha256": "e" * 64,
        "schema_sha256": "f" * 64,
        "scorer_sha256": "1" * 64,
        "workflow_sha256": "2" * 64,
        "workflow_component_sha256": {"runner.py": "3" * 64},
        "sources": [{"title": "dataset", "license": "license", "revision": "revision"}],
        "prompt_sha256": baseline_prompt_hash,
    }
    provider = {
        "requested_model": "nvidia_nim/google/gemma-4-31b-it",
        "expected_actual_model": "google/gemma-4-31b-it",
        "pricing_verified_on": "2026-08-16",
    }
    contract = {
        "target_sample_ids": [f"sample-{index:02d}" for index in range(40)],
        "provider": provider,
        "evaluation_mode": "benchmark",
        "evidence_kind": "live_quality",
        "fallback_enabled": False,
        "replay_enabled": False,
        "caps": {"max_requests": 40},
        "provenance": provenance,
    }
    summary_path = baseline / "summary.json"
    summary = {
        "observed_status": "complete",
        "record_count": 40,
        "target_count": 40,
        "requested_model": provider["requested_model"],
        "expected_actual_model": provider["expected_actual_model"],
        "actual_models": [provider["expected_actual_model"]],
        "provider_error_count": 0,
        "model_drift_count": 0,
        "live_call_performed": True,
        "provenance": provenance,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (baseline / "run-manifest.json").write_text(
        json.dumps({"contract": contract}), encoding="utf-8"
    )
    current_contract = {
        **contract,
        "provider": {**provider, "pricing_verified_on": "2026-08-17"},
        "provenance": {**provenance, "prompt_sha256": "9" * 64},
    }

    live_runner._require_baseline_release(baseline, current_contract)

    current_contract["provenance"]["config_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="prompt 외 계보"):
        live_runner._require_baseline_release(baseline, current_contract)


def test_full_cli_applies_local_prompt_without_sample_id(
    live_runner: ModuleType,
    project_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PromptApplied(Exception):
        pass

    local_data = tmp_path / "local-data"
    local_data.mkdir()
    prompt = local_data / "full-run-prompt.md"
    prompt.write_text("JSON 하나만 반환합니다.\n", encoding="utf-8")

    monkeypatch.setattr(live_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(live_runner, "load_project_env", lambda _root: None)
    monkeypatch.setattr(
        live_runner,
        "_require_approved_config",
        lambda _path: project_root / "configs/nvidia-nim-gemma4-baseline.yaml",
    )

    def assert_prompt_applied(changed, _config_path) -> None:
        assert changed.paths.prompt == "local-data/full-run-prompt.md"
        raise PromptApplied

    monkeypatch.setattr(live_runner, "_require_approved_provider", assert_prompt_applied)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nvidia_nim.py",
            *"--config configs/nvidia-nim-gemma4-baseline.yaml --live ".split(),
            "--prompt",
            str(prompt),
            "--baseline-run",
            "unused-baseline",
            *"--max-requests 40 --max-input-tokens 800000 ".split(),
            *"--max-output-tokens 20000 --max-cost-usd 0.01 ".split(),
            *"--max-wall-seconds 7200 --max-retries 0".split(),
            "--catalog-verified-on",
            live_runner.date.today().isoformat(),
            "--pricing-verified-on",
            live_runner.date.today().isoformat(),
        ],
    )

    with pytest.raises(PromptApplied):
        live_runner.main()


def test_full_local_prompt_requires_baseline(
    live_runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nvidia_nim.py",
            *"--live --prompt local-data/prompt.md ".split(),
            *"--max-requests 40 --max-input-tokens 800000 ".split(),
            *"--max-output-tokens 20000 --max-cost-usd 0.01 ".split(),
            *"--max-wall-seconds 7200 --max-retries 0".split(),
            "--catalog-verified-on",
            live_runner.date.today().isoformat(),
            "--pricing-verified-on",
            live_runner.date.today().isoformat(),
        ],
    )

    with pytest.raises(SystemExit):
        live_runner.main()


def test_live_run_preserves_and_verifies_exact_prompt(
    live_runner: ModuleType,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.md"
    snapshot = tmp_path / "prompt.md"
    source.write_text("JSON 하나만 반환합니다.\n", encoding="utf-8")
    expected = live_runner._sha256_file(source)

    live_runner._preserve_prompt(
        source,
        snapshot,
        expected_sha256=expected,
        resume=False,
    )
    assert snapshot.read_bytes() == source.read_bytes()

    snapshot.write_text("변조됨\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        live_runner._preserve_prompt(
            source,
            snapshot,
            expected_sha256=expected,
            resume=True,
        )


def test_terminal_provider_call_is_appended_once(live_runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    response = {"sample_id": "one", "provider_status": "provider_response_received"}
    terminal = {"sample_id": "one", "provider_status": "provider_error"}

    previous = live_runner._append_call_once(path, response, None)
    previous = live_runner._append_call_once(path, response, previous)
    live_runner._append_call_once(path, terminal, previous)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_known_live_validation_error_is_short(
    live_runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> int:
        raise ValueError("승인 상한이 다릅니다")

    monkeypatch.setattr(live_runner, "main", fail)

    assert live_runner.cli() == 2
    captured = capsys.readouterr()
    assert "NVIDIA NIM live 실행 차단: 승인 상한이 다릅니다" in captured.err
    assert "Traceback" not in captured.err
