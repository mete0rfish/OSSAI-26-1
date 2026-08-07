from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.live_execution import LiveExecutionError


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
    )

    assert status == "inconclusive"
    assert observed_status == "complete"


def test_full_run_keeps_observed_status(live_runner: ModuleType) -> None:
    status, observed_status = live_runner._classify_run_status(
        probe_only=False,
        blocked=False,
        complete=True,
        provider_error_count=0,
        model_drift_count=0,
    )

    assert status == observed_status == "complete"


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

    sealed = list(canonical)
    sealed[0] = sealed[0].model_copy(update={"split": "sealed_test"})
    with pytest.raises(ValueError, match="sealed_test"):
        live_runner._require_approved_case_copy(
            canonical_cases=canonical,
            local_cases=sealed,
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
        require_clean_git=False,
    )
    assert exploratory["git_clean"] is False

    with pytest.raises(RuntimeError, match="전체 품질 실행"):
        live_runner._build_provenance(
            settings=settings,
            config_path=project_root / "configs/nvidia-nim.yaml",
            cases=[case],
            input_manifest={"sample_ids": [case.sample_id]},
            catalog_verified_on=live_runner.date.today(),
            require_clean_git=True,
        )


def test_probe_prompt_must_be_an_existing_local_data_file(
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

    changed = live_runner._with_probe_prompt(settings, prompt)
    assert changed.paths.prompt == "local-data/my-prompt.md"

    outside = tmp_path / "outside.md"
    outside.write_text("허용하지 않는 위치\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local-data"):
        live_runner._with_probe_prompt(settings, outside)
