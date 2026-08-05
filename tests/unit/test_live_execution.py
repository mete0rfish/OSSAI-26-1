from pathlib import Path

import pytest

from verifiable_ai_workflow.live_execution import (
    LiveBudget,
    LiveBudgetCaps,
    LiveBudgetState,
    LiveExecutionError,
    require_canonical_project_file,
)


def test_canonical_project_file_rejects_external_and_symlink_config(tmp_path: Path) -> None:
    root = tmp_path / "project"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    canonical = config_dir / "live.yaml"
    canonical.write_text("provider: approved\n", encoding="utf-8")
    external = tmp_path / "external.yaml"
    external.write_text("provider: attacker\n", encoding="utf-8")

    resolved = require_canonical_project_file(
        root,
        "configs/live.yaml",
        "configs/live.yaml",
    )
    assert resolved == canonical
    with pytest.raises(LiveExecutionError, match="canonical"):
        require_canonical_project_file(root, external, "configs/live.yaml")

    canonical.unlink()
    canonical.symlink_to(external)
    with pytest.raises(LiveExecutionError, match="symlink"):
        require_canonical_project_file(root, canonical, "configs/live.yaml")


@pytest.mark.parametrize("field", ["max_cost_usd", "max_wall_seconds"])
def test_live_budget_caps_reject_non_finite_limits(field: str) -> None:
    values = {
        "max_requests": 1,
        "max_attempts": 1,
        "max_input_tokens": 100,
        "max_output_tokens": 50,
        "max_cost_usd": 0.1,
        "max_wall_seconds": 60,
    }
    values[field] = float("inf")

    with pytest.raises(ValueError):
        LiveBudgetCaps(**values)


def test_interrupted_attempt_keeps_reservation_after_resume() -> None:
    caps = LiveBudgetCaps(
        max_requests=2,
        max_attempts=2,
        max_input_tokens=200,
        max_output_tokens=100,
        max_cost_usd=0.2,
        max_wall_seconds=60,
    )
    first = LiveBudget(caps)
    first.reserve_attempt(
        sample_id="sample-1",
        request_number=None,
        reserved_input_tokens=100,
        reserved_output_tokens=50,
        reserved_cost_usd=0.1,
    )
    persisted = first.state.persisted_dict()

    resumed = LiveBudget(caps, state=LiveBudgetState.model_validate(persisted))
    resumed.recover_interrupted_attempts()

    assert resumed.state.attempts[0].status == "interrupted"
    assert resumed.state.charged_input_tokens == 100
    assert resumed.state.charged_output_tokens == 50
    assert resumed.state.charged_cost_usd == 0.1


def test_actual_usage_cannot_exceed_per_attempt_reservation() -> None:
    caps = LiveBudgetCaps(
        max_requests=1,
        max_attempts=1,
        max_input_tokens=1000,
        max_output_tokens=1000,
        max_cost_usd=1.0,
        max_wall_seconds=60,
    )
    budget = LiveBudget(caps)
    attempt = budget.reserve_attempt(
        sample_id="sample-1",
        request_number=None,
        reserved_input_tokens=100,
        reserved_output_tokens=50,
        reserved_cost_usd=0.1,
    )

    violations = budget.complete_attempt(
        attempt.attempt_number,
        input_tokens=101,
        output_tokens=51,
        actual_cost_usd=0.11,
    )

    assert violations == [
        "actual_input_tokens_exceeded_attempt_reservation",
        "actual_output_tokens_exceeded_attempt_reservation",
        "actual_cost_usd_exceeded_attempt_reservation",
    ]
