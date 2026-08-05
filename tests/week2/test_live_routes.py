from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from verifiable_ai_workflow.comparison import compare_routes
from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.live_provider_comparison import (
    LiveRoute,
    Week2LiveConfig,
    Week2LiveError,
    WholeRunCaps,
    attach_persistent_provider_budget,
    build_live_comparison_contract,
    default_provider_factory,
    enforce_live_comparison_requirements,
    load_week2_live_config,
    run_week2_live,
)
from verifiable_ai_workflow.schemas import EvaluationResult, PreparedDocument, PreparedPage

TODAY = date.today().isoformat()


@pytest.mark.parametrize("field", ["max_cost_usd", "max_wall_seconds"])
def test_whole_run_caps_reject_non_finite_limits(field: str) -> None:
    values = {
        "max_requests": 80,
        "max_input_tokens": 1_600_000,
        "max_output_tokens": 40_000,
        "max_retries": 0,
        "max_cost_usd": 0.40,
        "max_wall_seconds": 3600,
    }
    values[field] = float("inf")

    with pytest.raises(ValidationError):
        WholeRunCaps(**values)


def _write_prepared_documents(project_root: Path, target: Path) -> None:
    cases = build_cases(project_root / "data/cases/week-01-aihub.yaml")
    page_counts = {
        "MI2_240819_TY1_0012": 9,
        "MI2_240725_TY2_0002": 3,
    }
    text_by_page: dict[tuple[str, int], list[str]] = {}
    for case in cases:
        if case.expected.abstained:
            continue
        for page in case.expected.pages:
            text_by_page.setdefault((case.document_id, page), []).append(case.expected.answer)

    for document_id, page_count in page_counts.items():
        document_root = target / document_id
        pages: list[PreparedPage] = []
        for page_number in range(1, page_count + 1):
            image = document_root / "pages" / f"page-{page_number:04}.png"
            model_image = document_root / "model-pages" / f"page-{page_number:04}.jpg"
            text = document_root / "text" / f"page-{page_number:04}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            model_image.parent.mkdir(parents=True, exist_ok=True)
            text.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"png")
            model_image.write_bytes(b"jpeg")
            text.write_text(
                "\n".join(text_by_page.get((document_id, page_number), ["근거 없음"])),
                encoding="utf-8",
            )
            pages.append(
                PreparedPage(
                    page_number=page_number,
                    image_path=image.relative_to(document_root).as_posix(),
                    model_image_path=model_image.relative_to(document_root).as_posix(),
                    text_path=text.relative_to(document_root).as_posix(),
                )
            )
        manifest = PreparedDocument(
            document_id=document_id,
            source_file=f"{document_id}.pdf",
            source_sha256=("a" if document_id.endswith("0012") else "b") * 64,
            total_pages=page_count,
            render_dpi=150,
            pages=pages,
        )
        (document_root / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )


def _live_config(project_root: Path, tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    _write_prepared_documents(project_root, prepared)
    payload = yaml.safe_load(
        (project_root / "configs/week-02-live.yaml").read_text(encoding="utf-8")
    )
    payload["paths"]["case_authoring"] = str(project_root / "data/cases/week-01-aihub.yaml")
    payload["paths"]["prepared_documents"] = str(prepared)
    payload["paths"]["prompt"] = str(project_root / "prompts/pdf-question-answer.md")
    payload["paths"]["output"] = str(tmp_path / "reports")
    for route in (payload["baseline_route"], payload["candidate_route"]):
        route["pricing_verified_on"] = TODAY
        route["billing_basis"] = "developer_program_free_endpoint"
        route["input_cost_per_token_usd"] = 0.0
        route["output_cost_per_token_usd"] = 0.0
        route["task_budget"] = {
            "max_requests": 40,
            "requests_per_minute": 10000,
            "max_retries": 0,
            "retry_initial_seconds": 0.001,
            "max_cost_usd": 0.01,
            "max_input_tokens_per_request": 1,
            "max_output_tokens_per_request": 1,
            "request_timeout_seconds": 1,
            "max_wall_seconds": 60,
        }
    path = tmp_path / "week-02-live.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def _caps() -> WholeRunCaps:
    return WholeRunCaps(
        max_requests=80,
        max_input_tokens=80,
        max_output_tokens=80,
        max_retries=0,
        max_cost_usd=0.02,
        max_wall_seconds=120,
    )


def _response_by_id(project_root: Path) -> dict[str, dict[str, Any]]:
    responses: dict[str, dict[str, Any]] = {}
    for case in build_cases(project_root / "data/cases/week-01-aihub.yaml"):
        responses[case.sample_id] = {
            "answer": case.expected.answer,
            "evidence": (
                []
                if case.expected.abstained
                else [
                    {
                        "evidence_id": f"{case.sample_id}#page={case.expected.pages[0]}",
                        "quote": case.expected.answer,
                        "page_number": case.expected.pages[0],
                    }
                ]
            ),
            "confidence": 0.9,
            "abstained": case.expected.abstained,
            "abstention_reason": (
                "문서에서 확인할 수 없습니다." if case.expected.abstained else None
            ),
            "tool_requests": [],
        }
    return responses


class FakeLiveProvider:
    evidence_kind = "live_quality"

    def __init__(
        self,
        route: LiveRoute,
        responses: dict[str, dict[str, Any]],
        *,
        error_sample: str | None = None,
        actual_model: str | None = None,
    ) -> None:
        self.route = route
        self.responses = responses
        self.error_sample = error_sample
        self.actual_model = actual_model or route.expected_actual_model
        self.last_call: dict[str, Any] | None = None
        self.attempt_count = 0
        self.max_wall_seconds = route.task_budget.request_timeout_seconds

    def generate(self, sample_id: str, messages: list[dict[str, Any]]) -> str:
        assert messages
        self.attempt_count += 1
        if sample_id == self.error_sample:
            raise TimeoutError("fake timeout")
        self.last_call = {
            "sample_id": sample_id,
            "requested_model": self.route.model,
            "actual_model": self.actual_model,
            "response_id": f"{self.route.provider_id}-{sample_id}",
            "latency_ms": 1.0,
            "input_tokens": 1,
            "output_tokens": 1,
            "estimated_max_cost_usd": 0.0,
            "retry_count": 0,
            "request_number": self.attempt_count,
            "attempt_number": self.attempt_count,
            "rate_limit_headers": {},
        }
        return json.dumps(self.responses[sample_id], ensure_ascii=False)


def _factory(
    project_root: Path,
    *,
    candidate_error: str | None = None,
    candidate_actual_model: str | None = None,
    seen: list[str] | None = None,
) -> Callable[[LiveRoute], FakeLiveProvider]:
    responses = _response_by_id(project_root)

    def build(route: LiveRoute) -> FakeLiveProvider:
        if seen is not None:
            seen.append(route.provider_id)
        return FakeLiveProvider(
            route,
            responses,
            error_sample=(candidate_error if route.provider_id == "google-ai-studio" else None),
            actual_model=(
                candidate_actual_model
                if route.provider_id == "google-ai-studio"
                else route.expected_actual_model
            ),
        )

    return build


def _keys() -> dict[str, str]:
    return {
        "NVIDIA_NIM_API_KEY": "fake-nvidia-key",
        "GEMINI_API_KEY": "fake-gemini-key",
    }


def test_config_rejects_recorded_route_and_shared_identity(project_root: Path) -> None:
    payload = yaml.safe_load(
        (project_root / "configs/week-02-live.yaml").read_text(encoding="utf-8")
    )
    payload["baseline_route"]["provider_id"] = "recorded"
    payload["baseline_route"]["model"] = "recorded/fixture"
    with pytest.raises(ValidationError, match="recorded"):
        Week2LiveConfig.model_validate(payload)

    payload = yaml.safe_load(
        (project_root / "configs/week-02-live.yaml").read_text(encoding="utf-8")
    )
    payload["candidate_route"]["provider_id"] = payload["baseline_route"]["provider_id"]
    with pytest.raises(ValidationError, match="provider_id"):
        Week2LiveConfig.model_validate(payload)

    payload = yaml.safe_load(
        (project_root / "configs/week-02-live.yaml").read_text(encoding="utf-8")
    )
    payload["candidate_route"]["api_key_env"] = payload["baseline_route"]["api_key_env"]
    with pytest.raises(ValidationError, match="api_key_env"):
        Week2LiveConfig.model_validate(payload)


def test_default_factory_maps_route_total_and_request_caps(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_week2_live_config(project_root / "configs/week-02-live.yaml")
    route = config.candidate_route
    monkeypatch.setenv(route.api_key_env, "fake-key")

    provider = default_provider_factory(route)

    assert provider.model == "gemini/gemini-3.5-flash-lite"
    assert provider.api_base == "https://generativelanguage.googleapis.com/v1beta"
    assert provider.structured_output == "json_schema"
    assert provider.expected_actual_model == route.expected_actual_model
    assert provider.max_input_tokens == (
        route.task_budget.max_requests * route.task_budget.max_input_tokens_per_request
    )
    assert provider.request_input_token_ceiling == (route.task_budget.max_input_tokens_per_request)
    assert provider.max_wall_seconds == route.task_budget.max_wall_seconds
    assert provider.request_timeout_seconds == (route.task_budget.request_timeout_seconds)


def test_default_provider_persists_attempt_reservation_before_network(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_week2_live_config(project_root / "configs/week-02-live.yaml")
    route = config.baseline_route
    monkeypatch.setenv(route.api_key_env, "fake-key")
    provider = default_provider_factory(route)
    budget_path = tmp_path / "baseline-budget.json"

    assert attach_persistent_provider_budget(provider, budget_path) is True
    provider.budget.reserve_attempt(
        sample_id="aihub-report-r01",
        request_number=None,
        reserved_input_tokens=route.task_budget.max_input_tokens_per_request,
        reserved_output_tokens=route.task_budget.max_output_tokens_per_request,
        reserved_cost_usd=route.request_cost_ceiling_usd,
    )

    persisted = json.loads(budget_path.read_text(encoding="utf-8"))
    assert persisted["attempts"][0]["status"] == "reserved"
    assert persisted["attempts"][0]["sample_id"] == "aihub-report-r01"


def test_missing_second_key_blocks_before_provider_factory(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config_path = _live_config(project_root, tmp_path)
    factory_calls: list[str] = []

    with pytest.raises(Week2LiveError, match="GEMINI_API_KEY"):
        run_week2_live(
            project_root,
            config_path=config_path,
            caps=_caps(),
            catalog_verified_on=TODAY,
            output_dir=tmp_path / "output",
            environ={"NVIDIA_NIM_API_KEY": "only-first-key"},
            provider_factory=_factory(project_root, seen=factory_calls),
            require_clean_git=False,
            clock=lambda: 0.0,
        )

    assert factory_calls == []


def test_same_key_value_blocks_before_provider_factory(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config_path = _live_config(project_root, tmp_path)
    factory_calls: list[str] = []

    with pytest.raises(Week2LiveError, match="서로 다른 API key"):
        run_week2_live(
            project_root,
            config_path=config_path,
            caps=_caps(),
            catalog_verified_on=TODAY,
            output_dir=tmp_path / "output",
            environ={
                "NVIDIA_NIM_API_KEY": "same-key",
                "GEMINI_API_KEY": "same-key",
            },
            provider_factory=_factory(project_root, seen=factory_calls),
            require_clean_git=False,
            clock=lambda: 0.0,
        )

    assert factory_calls == []


def test_whole_run_caps_block_before_provider_factory(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config_path = _live_config(project_root, tmp_path)
    factory_calls: list[str] = []
    caps = _caps().model_copy(update={"max_requests": 79})

    with pytest.raises(Week2LiveError, match="max_requests"):
        run_week2_live(
            project_root,
            config_path=config_path,
            caps=caps,
            catalog_verified_on=TODAY,
            output_dir=tmp_path / "output",
            environ=_keys(),
            provider_factory=_factory(project_root, seen=factory_calls),
            require_clean_git=False,
            clock=lambda: 0.0,
        )

    assert factory_calls == []


def test_fake_two_provider_run_writes_raw_results_and_provenance(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config_path = _live_config(project_root, tmp_path)
    output = tmp_path / "output"
    seen: list[str] = []

    execution = run_week2_live(
        project_root,
        config_path=config_path,
        caps=_caps(),
        catalog_verified_on=TODAY,
        output_dir=output,
        environ=_keys(),
        provider_factory=_factory(project_root, seen=seen),
        require_clean_git=False,
        clock=lambda: 0.0,
    )

    assert seen == ["nvidia-nim", "google-ai-studio"]
    assert execution.summary.automated_status == "pass"
    assert execution.summary.baseline_observation_count == 40
    assert execution.summary.candidate_observation_count == 40
    assert execution.summary.baseline_invalid_outputs == 0
    assert execution.summary.candidate_invalid_outputs == 0
    assert execution.summary.release_claim is False
    assert execution.comparison.classification_counts == {"unchanged": 40}
    assert execution.baseline_provenance.input_manifest_sha256 == (
        execution.candidate_provenance.input_manifest_sha256
    )
    observation = json.loads(
        (output / "baseline-observations.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert observation["raw_output"]
    assert observation["model_call"]["provider_id"] == "nvidia-nim"
    assert (output / "baseline-results.jsonl").is_file()
    assert (output / "candidate-results.jsonl").is_file()
    assert (output / "baseline-live-records.jsonl").is_file()
    assert (output / "candidate-live-records.jsonl").is_file()
    input_manifest = json.loads((output / "input-manifest.json").read_text(encoding="utf-8"))
    assert input_manifest["input_modality"] == "page_images_only"
    assert input_manifest["scoring_profile"] == "aihub-vqa-deterministic-v2"
    assert "page_texts" not in json.dumps(input_manifest)
    assert (output / "run-manifest.json").is_file()
    assert (output / "baseline-provenance.json").is_file()
    assert (output / "candidate-provenance.json").is_file()
    record = json.loads(
        (output / "candidate-live-records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["run_id"] == execution.summary.run_id
    assert record["trial_id"].endswith(":candidate:aihub-report-r01:trial-01")
    assert record["source_license"] == "AIHub 이용정책 적용"
    assert record["git_sha"]
    assert record["lockfile_sha256"]
    assert record["raw_model_output"]
    assert record["parsed_response"]
    run_manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["status"] == "pass"
    assert run_manifest["baseline_observation_count"] == 40
    assert run_manifest["candidate_observation_count"] == 40


def test_one_sample_probe_calls_each_provider_once_and_is_inconclusive(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config_path = _live_config(project_root, tmp_path)
    output = tmp_path / "probe"
    seen: list[str] = []
    caps = WholeRunCaps(
        max_requests=2,
        max_input_tokens=2,
        max_output_tokens=2,
        max_retries=0,
        max_cost_usd=0.02,
        max_wall_seconds=2,
    )

    execution = run_week2_live(
        project_root,
        config_path=config_path,
        caps=caps,
        catalog_verified_on=TODAY,
        output_dir=output,
        environ=_keys(),
        provider_factory=_factory(project_root, seen=seen),
        require_clean_git=False,
        probe_sample_id="aihub-report-r01",
        clock=lambda: 0.0,
    )

    assert seen == ["nvidia-nim", "google-ai-studio"]
    assert execution.summary.baseline_observation_count == 1
    assert execution.summary.candidate_observation_count == 1
    assert execution.summary.automated_status == "inconclusive"
    assert len((output / "baseline-live-records.jsonl").read_text().splitlines()) == 1
    assert len((output / "candidate-live-records.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    ("candidate_error", "candidate_actual_model", "reason_fragment"),
    [
        ("aihub-report-r01", None, "provider error"),
        (None, "wrong-model", "actual model"),
    ],
)
def test_provider_error_or_actual_model_mismatch_is_inconclusive(
    project_root: Path,
    tmp_path: Path,
    candidate_error: str | None,
    candidate_actual_model: str | None,
    reason_fragment: str,
) -> None:
    config_path = _live_config(project_root, tmp_path)

    execution = run_week2_live(
        project_root,
        config_path=config_path,
        caps=_caps(),
        catalog_verified_on=TODAY,
        output_dir=tmp_path / "output",
        environ=_keys(),
        provider_factory=_factory(
            project_root,
            candidate_error=candidate_error,
            candidate_actual_model=candidate_actual_model,
        ),
        require_clean_git=False,
        clock=lambda: 0.0,
    )

    assert execution.summary.automated_status == "inconclusive"
    assert any(reason_fragment in reason for reason in execution.summary.invalid_reasons)


def test_incomplete_canonical_coverage_is_inconclusive(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config_path = _live_config(project_root, tmp_path)
    output = tmp_path / "output"
    execution = run_week2_live(
        project_root,
        config_path=config_path,
        caps=_caps(),
        catalog_verified_on=TODAY,
        output_dir=output,
        environ=_keys(),
        provider_factory=_factory(project_root),
        require_clean_git=False,
        clock=lambda: 0.0,
    )
    baseline = [
        EvaluationResult.model_validate_json(line)
        for line in (output / "baseline-results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    candidate = [
        EvaluationResult.model_validate_json(line)
        for line in (output / "candidate-results.jsonl").read_text(encoding="utf-8").splitlines()
    ][:-1]
    config = load_week2_live_config(config_path)
    contract = build_live_comparison_contract(project_root, config)
    report = compare_routes(
        baseline,
        candidate,
        baseline_route=config.baseline_route.descriptor(),
        candidate_route=config.candidate_route.descriptor(),
        baseline_contract=contract,
        candidate_contract=contract,
    )

    guarded = enforce_live_comparison_requirements(
        report,
        baseline_results=baseline,
        candidate_results=candidate,
        baseline_route=config.baseline_route,
        candidate_route=config.candidate_route,
    )

    assert execution.summary.automated_status == "pass"
    assert guarded.automated_status == "inconclusive"
    assert any("coverage 불완전" in reason for reason in guarded.invalid_comparison_reasons)
