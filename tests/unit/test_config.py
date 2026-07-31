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
    assert settings.provider.model == "nvidia_nim/google/gemma-4-31b-it"
    assert settings.provider.structured_output == "prompt_only"
    assert settings.limits.max_requests == 40
    assert settings.limits.requests_per_minute < 40


def test_api_key_is_read_only_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_TASK_KEY", raising=False)
    with pytest.raises(ValueError, match="환경 변수"):
        require_api_key("TEST_TASK_KEY")

    monkeypatch.setenv("TEST_TASK_KEY", "secret")
    assert require_api_key("TEST_TASK_KEY") == "secret"
