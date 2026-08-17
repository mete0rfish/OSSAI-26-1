from pathlib import Path

import pytest

from verifiable_ai_workflow.config import load_settings, project_path, require_api_key


def test_settings_load_from_standalone_project(project_root: Path) -> None:
    settings = load_settings(project_root / "configs/week-01.yaml")

    assert settings.provider.kind == "recorded"
    assert settings.provider.api_key_env is None
    assert project_path(project_root, settings.paths.case_authoring).is_file()


def test_nvidia_nim_config_is_ready_for_live_batch(project_root: Path) -> None:
    settings = load_settings(project_root / "configs/nvidia-nim.yaml")

    assert settings.provider.kind == "litellm"
    assert (
        settings.provider.model
        == "nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    )
    assert (
        settings.provider.expected_actual_model
        == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    )
    assert settings.provider.billing_basis == "developer_program_free_endpoint"
    assert settings.provider.structured_output == "prompt_only"
    assert settings.provider.billing_basis == "developer_program_free_endpoint"
    assert settings.limits.max_requests == 40
    assert settings.limits.requests_per_minute < 40
    assert settings.limits.request_output_token_ceiling == 500
    assert settings.limits.request_output_token_ceiling < settings.limits.max_output_tokens


def test_gemma_prompt_candidate_has_separate_paths_and_same_live_model(
    project_root: Path,
) -> None:
    baseline = load_settings(project_root / "configs/nvidia-nim-gemma4-baseline.yaml")
    candidate = load_settings(project_root / "configs/nvidia-nim-gemma4.yaml")

    assert candidate.provider.model == baseline.provider.model
    assert candidate.provider.expected_actual_model == baseline.provider.expected_actual_model
    assert candidate.paths.cases == baseline.paths.cases
    assert candidate.paths.prepared_documents == baseline.paths.prepared_documents
    assert candidate.paths.prompt != baseline.paths.prompt
    assert candidate.paths.output != baseline.paths.output
    assert candidate.paths.recorded_responses is None
    assert not (project_root / "configs/week-01-gemma4.yaml").exists()


def test_api_key_is_read_only_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_TASK_KEY", raising=False)
    with pytest.raises(ValueError, match="환경 변수"):
        require_api_key("TEST_TASK_KEY")

    monkeypatch.setenv("TEST_TASK_KEY", "secret")
    assert require_api_key("TEST_TASK_KEY") == "secret"
