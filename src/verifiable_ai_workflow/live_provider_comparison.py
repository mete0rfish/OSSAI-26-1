"""Week 2의 동일 40건을 실제 두 provider로 제한 실행하는 경로."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from .comparison import (
    ComparisonContract,
    ComparisonReport,
    RouteDescriptor,
    compare_routes,
    sha256_file,
)
from .data.dataset import build_cases
from .evaluation.scoring import SCORING_PROFILE, score_observations
from .live_execution import LiveBudget, atomic_write_json
from .providers.litellm_provider import LiteLLMProvider
from .schemas import EvaluationCase, EvaluationResult, ModelObservation
from .workflow import run_cases

EXPECTED_WEEK2_SAMPLE_IDS = tuple(
    [f"aihub-report-r{index:02d}" for index in range(1, 33)]
    + [f"aihub-press-p{index:02d}" for index in range(1, 9)]
)
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9-]*$")


class Week2LiveError(RuntimeError):
    """live 실행 준비 또는 제한을 만족하지 못해 network 전에 차단됨."""


class ProviderCallFailed(RuntimeError):
    """secret이나 provider 원문 오류를 artifact에 복사하지 않는 호출 실패."""


class LiveModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouteTaskBudget(LiveModel):
    max_requests: int = Field(gt=0)
    requests_per_minute: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    retry_initial_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    max_input_tokens_per_request: int = Field(gt=0)
    max_output_tokens_per_request: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_wall_seconds: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def timeout_fits_route_wall(self) -> RouteTaskBudget:
        if self.request_timeout_seconds > self.max_wall_seconds:
            raise ValueError("요청 timeout은 route 전체 시간 상한보다 클 수 없습니다")
        return self


class LiveRoute(LiveModel):
    provider_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    expected_actual_model: str = Field(min_length=1)
    api_base: str | None
    api_key_env: str = Field(min_length=1)
    structured_output: Literal["json_schema", "prompt_only"]
    billing_basis: Literal["developer_program_free_endpoint", "per_token"]
    pricing_source_url: str = Field(min_length=1)
    pricing_verified_on: date
    input_cost_per_token_usd: float = Field(ge=0, allow_inf_nan=False)
    output_cost_per_token_usd: float = Field(ge=0, allow_inf_nan=False)
    task_budget: RouteTaskBudget

    @computed_field
    @property
    def request_cost_ceiling_usd(self) -> float:
        return (
            self.task_budget.max_input_tokens_per_request * self.input_cost_per_token_usd
            + self.task_budget.max_output_tokens_per_request * self.output_cost_per_token_usd
        )

    @model_validator(mode="after")
    def validate_live_route(self) -> LiveRoute:
        provider_id = self.provider_id.casefold()
        model = self.model.casefold()
        if not _PROVIDER_ID.fullmatch(self.provider_id):
            raise ValueError("provider_id는 소문자 영문·숫자·하이픈만 사용합니다")
        if "recorded" in provider_id or "fixture" in provider_id:
            raise ValueError("live route에 recorded provider를 사용할 수 없습니다")
        if model.startswith(("recorded", "fixture")):
            raise ValueError("live route에 recorded/fixture model을 사용할 수 없습니다")
        if not _ENV_NAME.fullmatch(self.api_key_env):
            raise ValueError("api_key_env는 환경 변수 이름이어야 합니다")
        if self.api_base is not None and not self.api_base.startswith(("https://", "http://")):
            raise ValueError("api_base는 http(s) URL 또는 null이어야 합니다")
        if not self.pricing_source_url.startswith("https://"):
            raise ValueError("pricing_source_url은 HTTPS 공식 문서여야 합니다")
        if self.billing_basis == "developer_program_free_endpoint":
            if self.input_cost_per_token_usd != 0 or self.output_cost_per_token_usd != 0:
                raise ValueError("무료 개발 endpoint의 token 단가는 0이어야 합니다")
        elif self.input_cost_per_token_usd <= 0 or self.output_cost_per_token_usd <= 0:
            raise ValueError("per_token route에는 양의 입력·출력 단가가 필요합니다")
        worst_case_cost = self.request_cost_ceiling_usd * self.task_budget.max_requests
        if worst_case_cost > self.task_budget.max_cost_usd + 1e-12:
            raise ValueError("route 요청별 최악 비용의 합이 task_budget.max_cost_usd를 넘습니다")
        return self

    def descriptor(self) -> RouteDescriptor:
        return RouteDescriptor(
            logical_model=f"{self.provider_id}:{self.model}",
            requested_model=self.model,
            expected_actual_model=self.expected_actual_model,
        )


class LivePaths(LiveModel):
    case_authoring: str
    prepared_documents: str
    prompt: str
    output: str


class Week2LiveConfig(LiveModel):
    artifact_schema_version: Literal[3]
    expected_sample_count: Literal[40]
    paths: LivePaths
    baseline_route: LiveRoute
    candidate_route: LiveRoute

    @model_validator(mode="after")
    def routes_are_independent_and_comparable(self) -> Week2LiveConfig:
        if self.baseline_route.provider_id == self.candidate_route.provider_id:
            raise ValueError("두 live route의 provider_id는 달라야 합니다")
        if self.baseline_route.api_key_env == self.candidate_route.api_key_env:
            raise ValueError("두 live route의 api_key_env는 달라야 합니다")
        if (
            self.baseline_route.task_budget.max_requests != self.expected_sample_count
            or self.candidate_route.task_budget.max_requests != self.expected_sample_count
        ):
            raise ValueError("각 route max_requests는 정확히 40이어야 합니다")
        if (
            self.baseline_route.task_budget.max_output_tokens_per_request
            != self.candidate_route.task_budget.max_output_tokens_per_request
        ):
            raise ValueError("두 route의 max output token 설정은 같아야 합니다")
        return self


class WholeRunCaps(LiveModel):
    max_requests: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    max_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    max_wall_seconds: float = Field(gt=0, allow_inf_nan=False)


class FileDigest(LiveModel):
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DocumentInputDigest(LiveModel):
    document_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_pages: int = Field(gt=0)
    model_pages: list[FileDigest] = Field(min_length=1)


class LiveInputManifest(LiveModel):
    input_modality: Literal["page_images_only"] = "page_images_only"
    scoring_profile: Literal["aihub-vqa-deterministic-v2"] = SCORING_PROFILE
    case_authoring_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_ids: tuple[str, ...]
    documents: list[DocumentInputDigest]

    @computed_field
    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(exclude={"sha256"}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class RouteBudgetSnapshot(LiveModel):
    max_requests: int
    reserved_requests: int
    reserved_input_tokens: int
    reserved_output_tokens: int
    reserved_retries: int
    reserved_cost_usd: float
    observed_attempts: int
    observed_retries: int
    observed_input_tokens: int
    observed_output_tokens: int
    observed_estimated_cost_usd: float
    elapsed_seconds: float
    violation: str | None


class GitProvenance(LiveModel):
    sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    clean: bool


class RouteProvenance(LiveModel):
    artifact_schema_version: Literal[3] = 3
    role: Literal["baseline", "candidate"]
    evidence_kind: Literal["live_quality"] = "live_quality"
    provider_id: str
    requested_model: str
    expected_actual_model: str
    api_base: str | None
    api_key_env: str
    billing_basis: Literal["developer_program_free_endpoint", "per_token"]
    pricing_source_url: str
    pricing_verified_on: date
    git: GitProvenance
    catalog_verified_on: date
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at_utc: datetime
    completed_at_utc: datetime
    expected_sample_count: Literal[40] = 40
    observation_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    provider_error_count: int = Field(ge=0)
    invalid_output_count: int = Field(ge=0)
    actual_model_mismatch_count: int = Field(ge=0)
    budget: RouteBudgetSnapshot
    raw_observations_file: str
    results_file: str
    live_records_file: str


class LiveSampleRecord(LiveModel):
    """한 sample의 live 입력·응답·평가와 실행 identity를 함께 보존한다."""

    artifact_schema_version: Literal[3] = 3
    evidence_kind: Literal["live_quality"] = "live_quality"
    run_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    trial_index: Literal[1] = 1
    route_role: Literal["baseline", "candidate"]
    sample_id: str
    family_id: str
    split: Literal["development", "validation", "challenge", "sealed_test"]
    risk_level: Literal["low", "medium", "high"]
    source_title: str
    source_license: str
    dataset_revision: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_clean: bool
    workflow_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lockfile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rubric_sha256: None = None
    output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_id: str
    requested_model: str
    expected_actual_model: str
    actual_model: str | None
    provider_status: Literal["success", "invalid_output", "provider_error", "blocked"]
    provider_raw_response: Any = None
    raw_model_output: Any = None
    parsed_response: dict[str, Any] | None
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    final_state: None = None
    scores: dict[str, float]
    scorer_reasons: dict[str, str]
    human_result: None = None
    judge_result: None = None
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    retry_count: int | None = Field(default=None, ge=0)
    attempt_trace: list[dict[str, Any]] = Field(default_factory=list)
    api_error: str | None
    started_at_utc: datetime
    completed_at_utc: datetime


class LiveWeek2Summary(LiveModel):
    artifact_schema_version: Literal[3] = 3
    evidence_kind: Literal["live_quality"] = "live_quality"
    run_id: str = Field(min_length=1)
    automated_status: Literal["pass", "fail", "inconclusive"]
    automated_reason: str
    invalid_reasons: list[str]
    expected_sample_count: Literal[40] = 40
    baseline_observation_count: int
    candidate_observation_count: int
    baseline_provider_errors: int
    candidate_provider_errors: int
    baseline_invalid_outputs: int
    candidate_invalid_outputs: int
    input_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_claim: Literal[False] = False


class LiveWeek2Execution(LiveModel):
    summary: LiveWeek2Summary
    comparison: ComparisonReport
    baseline_provenance: RouteProvenance
    candidate_provenance: RouteProvenance


class LiveProvider(Protocol):
    evidence_kind: Literal["live_quality"]
    last_call: dict[str, Any] | None

    def generate(self, sample_id: str, messages: list[dict[str, Any]]) -> Any: ...


ProviderFactory = Callable[[LiveRoute], LiveProvider]


def load_week2_live_config(path: str | Path) -> Week2LiveConfig:
    return Week2LiveConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _rooted(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _collect_git_provenance(project_root: Path) -> GitProvenance:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return GitProvenance(sha=sha, clean=not status)


def _validate_expected_cases(cases: list[EvaluationCase]) -> None:
    actual = tuple(case.sample_id for case in cases)
    if actual != EXPECTED_WEEK2_SAMPLE_IDS:
        missing = sorted(set(EXPECTED_WEEK2_SAMPLE_IDS) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_WEEK2_SAMPLE_IDS))
        raise Week2LiveError(
            "Week 2 live 입력은 canonical 40건과 정확히 같아야 합니다: "
            f"count={len(actual)}, missing={missing}, extra={extra}"
        )


def build_live_input_manifest(
    project_root: str | Path,
    config: Week2LiveConfig,
    cases: list[EvaluationCase],
) -> LiveInputManifest:
    root = Path(project_root)
    _validate_expected_cases(cases)
    prepared_root = _rooted(root, config.paths.prepared_documents)
    documents: list[DocumentInputDigest] = []
    for document_id in sorted({case.document_id for case in cases}):
        manifest_path = prepared_root / document_id / "manifest.json"
        if not manifest_path.is_file():
            raise Week2LiveError(f"전처리 manifest가 없습니다: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_pages: list[FileDigest] = []
        for page in payload["pages"]:
            model_path = manifest_path.parent / page["model_image_path"]
            if not model_path.is_file():
                raise Week2LiveError(f"전처리 model image가 없습니다: {document_id}")
            model_pages.append(
                FileDigest(
                    relative_path=model_path.relative_to(prepared_root).as_posix(),
                    sha256=sha256_file(model_path),
                )
            )
        documents.append(
            DocumentInputDigest(
                document_id=document_id,
                source_sha256=payload["source_sha256"],
                manifest_sha256=sha256_file(manifest_path),
                total_pages=payload["total_pages"],
                model_pages=model_pages,
            )
        )
    return LiveInputManifest(
        case_authoring_sha256=sha256_file(_rooted(root, config.paths.case_authoring)),
        prompt_sha256=sha256_file(_rooted(root, config.paths.prompt)),
        output_schema_sha256=sha256_file(root / "src/verifiable_ai_workflow/schemas/models.py"),
        scorer_sha256=sha256_file(root / "src/verifiable_ai_workflow/evaluation/scoring.py"),
        sample_ids=EXPECTED_WEEK2_SAMPLE_IDS,
        documents=documents,
    )


def build_live_comparison_contract(
    project_root: str | Path,
    config: Week2LiveConfig,
) -> ComparisonContract:
    root = Path(project_root)
    return ComparisonContract(
        scoring_profile=SCORING_PROFILE,
        dataset_sha256=sha256_file(_rooted(root, config.paths.case_authoring)),
        prompt_sha256=sha256_file(_rooted(root, config.paths.prompt)),
        output_schema_sha256=sha256_file(root / "src/verifiable_ai_workflow/schemas/models.py"),
        scorer_sha256=sha256_file(root / "src/verifiable_ai_workflow/evaluation/scoring.py"),
        lockfile_sha256=sha256_file(root / "uv.lock"),
        temperature=0.0,
        max_output_tokens=(config.baseline_route.task_budget.max_output_tokens_per_request),
    )


def resolve_live_keys(
    config: Week2LiveConfig,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = os.environ if environ is None else environ
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for route in (config.baseline_route, config.candidate_route):
        value = values.get(route.api_key_env)
        if not value:
            missing.append(route.api_key_env)
        else:
            resolved[route.provider_id] = value
    if missing:
        raise Week2LiveError(
            "두 provider key를 모두 확인하기 전에는 network를 시작하지 않습니다: "
            + ", ".join(sorted(missing))
        )
    baseline_key = resolved[config.baseline_route.provider_id]
    candidate_key = resolved[config.candidate_route.provider_id]
    if hmac.compare_digest(baseline_key, candidate_key):
        raise Week2LiveError("두 provider는 서로 다른 API key를 사용해야 합니다")
    return resolved


def validate_whole_run_caps(config: Week2LiveConfig, caps: WholeRunCaps) -> None:
    validate_whole_run_caps_for_sample_count(
        config,
        caps,
        samples_per_route=config.expected_sample_count,
    )


def validate_whole_run_caps_for_sample_count(
    config: Week2LiveConfig,
    caps: WholeRunCaps,
    *,
    samples_per_route: int,
) -> None:
    if samples_per_route <= 0 or samples_per_route > config.expected_sample_count:
        raise Week2LiveError("route별 실행 sample 수는 1~40이어야 합니다")
    routes = (config.baseline_route, config.candidate_route)
    required = {
        "max_requests": samples_per_route * len(routes),
        "max_input_tokens": sum(
            samples_per_route * route.task_budget.max_input_tokens_per_request for route in routes
        ),
        "max_output_tokens": sum(
            samples_per_route * route.task_budget.max_output_tokens_per_request for route in routes
        ),
        "max_retries": sum(samples_per_route * route.task_budget.max_retries for route in routes),
        "max_cost_usd": sum(samples_per_route * route.request_cost_ceiling_usd for route in routes),
        "max_wall_seconds": sum(
            min(
                route.task_budget.max_wall_seconds,
                samples_per_route * route.task_budget.request_timeout_seconds,
            )
            for route in routes
        ),
    }
    insufficient = [
        f"{name}={getattr(caps, name)} < required={value}"
        for name, value in required.items()
        if getattr(caps, name) + 1e-12 < value
    ]
    if insufficient:
        raise Week2LiveError(
            "CLI 전체 cap이 두 route 예산보다 작습니다: " + "; ".join(insufficient)
        )


class _WholeRunGuard:
    def __init__(
        self,
        caps: WholeRunCaps,
        *,
        clock: Callable[[], float],
    ) -> None:
        self.caps = caps
        self.clock = clock
        self.started_at = clock()
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.retries = 0
        self.cost_usd = 0.0

    def reserve(self, route: LiveRoute) -> float:
        budget = route.task_budget
        proposed = {
            "requests": self.requests + 1,
            "input_tokens": self.input_tokens + budget.max_input_tokens_per_request,
            "output_tokens": self.output_tokens + budget.max_output_tokens_per_request,
            "retries": self.retries + budget.max_retries,
            "cost_usd": self.cost_usd + route.request_cost_ceiling_usd,
        }
        limits = {
            "requests": self.caps.max_requests,
            "input_tokens": self.caps.max_input_tokens,
            "output_tokens": self.caps.max_output_tokens,
            "retries": self.caps.max_retries,
            "cost_usd": self.caps.max_cost_usd,
        }
        exceeded = [name for name, value in proposed.items() if value > limits[name] + 1e-12]
        remaining = self.caps.max_wall_seconds - (self.clock() - self.started_at)
        if exceeded or remaining <= 0:
            reason = ", ".join(exceeded) if exceeded else "wall_seconds"
            raise Week2LiveError(f"CLI 전체 cap을 넘기기 전에 차단했습니다: {reason}")
        self.requests = int(proposed["requests"])
        self.input_tokens = int(proposed["input_tokens"])
        self.output_tokens = int(proposed["output_tokens"])
        self.retries = int(proposed["retries"])
        self.cost_usd = float(proposed["cost_usd"])
        return remaining


class _BudgetedRouteProvider:
    evidence_kind: Literal["live_quality"] = "live_quality"

    def __init__(
        self,
        route: LiveRoute,
        provider: LiveProvider,
        whole_guard: _WholeRunGuard,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if getattr(provider, "evidence_kind", None) != "live_quality":
            raise Week2LiveError(f"{route.provider_id} live route가 실제 API provider가 아닙니다")
        self.route = route
        self.provider = provider
        self.whole_guard = whole_guard
        self.clock = clock
        self.started_at = clock()
        self.last_call: dict[str, Any] | None = None
        self.reserved_requests = 0
        self.reserved_input_tokens = 0
        self.reserved_output_tokens = 0
        self.reserved_retries = 0
        self.reserved_cost_usd = 0.0
        self.observed_attempts = 0
        self.observed_retries = 0
        self.observed_input_tokens = 0
        self.observed_output_tokens = 0
        self.observed_estimated_cost_usd = 0.0
        self.violation: str | None = None

    def generate(self, sample_id: str, messages: list[dict[str, Any]]) -> Any:
        budget = self.route.task_budget
        route_remaining = budget.max_wall_seconds - (self.clock() - self.started_at)
        if self.violation is not None:
            raise Week2LiveError(f"이전 budget violation 뒤 route를 차단했습니다: {self.violation}")
        if self.reserved_requests >= budget.max_requests:
            raise Week2LiveError("route 요청 수 상한을 넘기기 전에 차단했습니다")
        if route_remaining <= 0:
            raise Week2LiveError("route 전체 시간 상한을 넘기기 전에 차단했습니다")
        whole_remaining = self.whole_guard.reserve(self.route)

        self.reserved_requests += 1
        self.reserved_input_tokens += budget.max_input_tokens_per_request
        self.reserved_output_tokens += budget.max_output_tokens_per_request
        self.reserved_retries += budget.max_retries
        self.reserved_cost_usd += self.route.request_cost_ceiling_usd
        if self.reserved_cost_usd > budget.max_cost_usd + 1e-12:
            raise Week2LiveError("route 비용 상한을 넘기기 전에 차단했습니다")

        timeout = min(
            budget.request_timeout_seconds,
            route_remaining,
            whole_remaining,
        )
        if hasattr(self.provider, "request_timeout_seconds"):
            self.provider.request_timeout_seconds = timeout
        elif hasattr(self.provider, "max_wall_seconds"):
            self.provider.max_wall_seconds = timeout
        if hasattr(self.provider, "last_call"):
            self.provider.last_call = None
        before_attempts = getattr(self.provider, "attempt_count", None)
        try:
            raw_output = self.provider.generate(sample_id, messages)
        except Exception as exc:
            self._capture_call(before_attempts)
            self._check_observed_budget()
            raise ProviderCallFailed(
                f"{self.route.provider_id} 호출 실패: {type(exc).__name__}"
            ) from exc
        self._capture_call(before_attempts)
        self._check_observed_budget()
        return raw_output

    def _capture_call(self, before_attempts: int | None) -> None:
        call = getattr(self.provider, "last_call", None)
        if call is not None:
            self.last_call = {
                **dict(call),
                "provider_id": self.route.provider_id,
                "api_base": self.route.api_base,
            }
            self.observed_retries += int(call.get("retry_count") or 0)
            self.observed_input_tokens += int(call.get("input_tokens") or 0)
            self.observed_output_tokens += int(call.get("output_tokens") or 0)
            self.observed_estimated_cost_usd += float(call.get("estimated_max_cost_usd") or 0.0)
        else:
            self.last_call = None
        after_attempts = getattr(self.provider, "attempt_count", None)
        if before_attempts is not None and after_attempts is not None:
            attempt_delta = max(0, after_attempts - before_attempts)
            self.observed_attempts += attempt_delta
            if call is None:
                self.observed_retries += max(0, attempt_delta - 1)
        elif call is not None:
            self.observed_attempts += 1 + int(call.get("retry_count") or 0)

    def _check_observed_budget(self) -> None:
        budget = self.route.task_budget
        problems: list[str] = []
        if self.observed_retries > self.reserved_retries:
            problems.append("retry")
        if self.observed_input_tokens > self.reserved_input_tokens:
            problems.append("input_tokens")
        if self.observed_output_tokens > self.reserved_output_tokens:
            problems.append("output_tokens")
        if self.observed_estimated_cost_usd > budget.max_cost_usd + 1e-12:
            problems.append("cost")
        if problems:
            self.violation = ",".join(problems)
            if self.last_call is not None:
                self.last_call["budget_violation"] = self.violation

    def snapshot(self) -> RouteBudgetSnapshot:
        return RouteBudgetSnapshot(
            max_requests=self.route.task_budget.max_requests,
            reserved_requests=self.reserved_requests,
            reserved_input_tokens=self.reserved_input_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            reserved_retries=self.reserved_retries,
            reserved_cost_usd=round(self.reserved_cost_usd, 8),
            observed_attempts=self.observed_attempts,
            observed_retries=self.observed_retries,
            observed_input_tokens=self.observed_input_tokens,
            observed_output_tokens=self.observed_output_tokens,
            observed_estimated_cost_usd=round(
                self.observed_estimated_cost_usd,
                8,
            ),
            elapsed_seconds=round(self.clock() - self.started_at, 6),
            violation=self.violation,
        )


def default_provider_factory(route: LiveRoute) -> LiveProvider:
    budget = route.task_budget
    return LiteLLMProvider(
        model=route.model,
        api_key_env=route.api_key_env,
        api_base=route.api_base,
        structured_output=route.structured_output,
        max_requests=budget.max_requests,
        requests_per_minute=budget.requests_per_minute,
        max_retries=budget.max_retries,
        retry_initial_seconds=budget.retry_initial_seconds,
        max_cost_usd=budget.max_cost_usd,
        max_input_tokens=(budget.max_requests * budget.max_input_tokens_per_request),
        max_output_tokens=(budget.max_requests * budget.max_output_tokens_per_request),
        max_wall_seconds=budget.max_wall_seconds,
        expected_actual_model=route.expected_actual_model,
        max_attempts=budget.max_requests * (budget.max_retries + 1),
        request_input_token_ceiling=budget.max_input_tokens_per_request,
        request_output_token_ceiling=budget.max_output_tokens_per_request,
        request_timeout_seconds=budget.request_timeout_seconds,
        input_cost_per_token_usd=route.input_cost_per_token_usd,
        output_cost_per_token_usd=route.output_cost_per_token_usd,
    )


def attach_persistent_provider_budget(provider: LiveProvider, path: Path) -> bool:
    """LiteLLM attempt 예약을 network 전에 route별 파일에 저장한다."""

    budget = getattr(provider, "budget", None)
    if not isinstance(budget, LiveBudget):
        return False
    atomic_write_json(path, budget.state.persisted_dict())
    budget.set_on_change(
        lambda state: atomic_write_json(
            path,
            state.persisted_dict(),
        )
    )
    return True


def attach_provider_response_journal(provider: LiveProvider, path: Path) -> bool:
    """HTTP 응답 원문을 budget 정산보다 먼저 append+fsync한다."""

    setter = getattr(provider, "set_on_response_received", None)
    if not callable(setter):
        return False
    setter(lambda record: _append_jsonl_fsync(path, record))
    return True


def _write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_jsonl_fsync(path: Path, row: BaseModel | Mapping[str, Any]) -> None:
    payload = (
        row.model_dump_json()
        if isinstance(row, BaseModel)
        else json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), default=str)
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_persisted_observations(
    path: Path,
    observations: list[ModelObservation],
) -> None:
    persisted = [
        ModelObservation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if persisted != observations:
        raise Week2LiveError(f"{path.name}의 즉시 저장 observation이 메모리 결과와 다릅니다")


def _actual_model_mismatch_count(
    route: LiveRoute,
    results: list[EvaluationResult],
) -> int:
    count = 0
    for result in results:
        if result.provider_status == "provider_error":
            continue
        actual = result.model_call.get("actual_model") if result.model_call else None
        if actual != route.expected_actual_model:
            count += 1
    return count


def _coverage_reasons(
    label: str,
    results: list[EvaluationResult],
    route: LiveRoute,
) -> list[str]:
    reasons: list[str] = []
    ids = [result.sample_id for result in results]
    if tuple(ids) != EXPECTED_WEEK2_SAMPLE_IDS:
        reasons.append(f"{label} canonical 40건 coverage 불완전")
    provider_errors = sum(result.provider_status == "provider_error" for result in results)
    if provider_errors:
        reasons.append(f"{label} provider error {provider_errors}건")
    mismatch = _actual_model_mismatch_count(route, results)
    if mismatch:
        reasons.append(f"{label} actual model 불일치 또는 미보고 {mismatch}건")
    return reasons


def enforce_live_comparison_requirements(
    report: ComparisonReport,
    *,
    baseline_results: list[EvaluationResult],
    candidate_results: list[EvaluationResult],
    baseline_route: LiveRoute,
    candidate_route: LiveRoute,
    extra_reasons: list[str] | None = None,
) -> ComparisonReport:
    reasons = list(report.invalid_comparison_reasons)
    reasons.extend(_coverage_reasons("baseline", baseline_results, baseline_route))
    reasons.extend(_coverage_reasons("candidate", candidate_results, candidate_route))
    reasons.extend(extra_reasons or [])
    reasons = list(dict.fromkeys(reasons))
    if not reasons:
        return report
    return report.model_copy(
        update={
            "invalid_comparison_reasons": reasons,
            "automated_status": "inconclusive",
            "automated_reason": (
                "provider 오류, model identity, coverage, 예산 또는 입력 고정 조건을 "
                "만족하지 못해 비교할 수 없습니다."
            ),
        }
    )


def _route_provenance(
    *,
    role: Literal["baseline", "candidate"],
    route: LiveRoute,
    observations: list[ModelObservation],
    results: list[EvaluationResult],
    git: GitProvenance,
    catalog_verified_on: date,
    config_sha256: str,
    contract: ComparisonContract,
    input_manifest: LiveInputManifest,
    started_at: datetime,
    completed_at: datetime,
    budget: RouteBudgetSnapshot,
    live_records_file: str,
) -> RouteProvenance:
    return RouteProvenance(
        role=role,
        provider_id=route.provider_id,
        requested_model=route.model,
        expected_actual_model=route.expected_actual_model,
        api_base=route.api_base,
        api_key_env=route.api_key_env,
        billing_basis=route.billing_basis,
        pricing_source_url=route.pricing_source_url,
        pricing_verified_on=route.pricing_verified_on,
        git=git,
        catalog_verified_on=catalog_verified_on,
        config_sha256=config_sha256,
        comparison_contract_sha256=contract.sha256,
        input_manifest_sha256=input_manifest.sha256,
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        observation_count=len(observations),
        result_count=len(results),
        provider_error_count=sum(result.provider_status == "provider_error" for result in results),
        invalid_output_count=sum(result.provider_status == "invalid_output" for result in results),
        actual_model_mismatch_count=_actual_model_mismatch_count(route, results),
        budget=budget,
        raw_observations_file=f"{role}-observations.jsonl",
        results_file=f"{role}-results.jsonl",
        live_records_file=live_records_file,
    )


def _build_live_sample_records(
    *,
    run_id: str,
    role: Literal["baseline", "candidate"],
    route: LiveRoute,
    cases: list[EvaluationCase],
    observations: list[ModelObservation],
    results: list[EvaluationResult],
    input_manifest: LiveInputManifest,
    contract: ComparisonContract,
    config_sha256: str,
    git: GitProvenance,
    started_at: datetime,
    completed_at: datetime,
) -> list[LiveSampleRecord]:
    cases_by_id = {case.sample_id: case for case in cases}
    observations_by_id = {observation.sample_id: observation for observation in observations}
    document_manifests = {
        document.document_id: document.manifest_sha256 for document in input_manifest.documents
    }
    records: list[LiveSampleRecord] = []
    for result in results:
        case = cases_by_id[result.sample_id]
        observation = observations_by_id[result.sample_id]
        call = observation.model_call or {}
        parsed = result.output.model_dump(mode="json") if result.output is not None else None
        records.append(
            LiveSampleRecord(
                run_id=run_id,
                trial_id=f"{run_id}:{role}:{result.sample_id}:trial-01",
                route_role=role,
                sample_id=result.sample_id,
                family_id=result.family_id,
                split=case.split,
                risk_level=case.risk_level,
                source_title=case.source.title,
                source_license=case.source.license,
                dataset_revision=case.source.revision,
                source_manifest_sha256=document_manifests[case.document_id],
                dataset_manifest_sha256=input_manifest.sha256,
                git_sha=git.sha,
                git_clean=git.clean,
                workflow_config_sha256=config_sha256,
                lockfile_sha256=contract.lockfile_sha256,
                prompt_sha256=contract.prompt_sha256,
                output_schema_sha256=contract.output_schema_sha256,
                scorer_sha256=contract.scorer_sha256,
                provider_id=route.provider_id,
                requested_model=route.model,
                expected_actual_model=route.expected_actual_model,
                actual_model=call.get("actual_model"),
                provider_status=result.provider_status,
                provider_raw_response=call.get("raw_response", observation.raw_output),
                raw_model_output=observation.raw_output,
                parsed_response=parsed,
                scores=result.scores,
                scorer_reasons=result.reasons,
                latency_ms=call.get("latency_ms"),
                input_tokens=call.get("input_tokens"),
                output_tokens=call.get("output_tokens"),
                estimated_cost_usd=call.get("actual_cost_usd"),
                retry_count=call.get("retry_count"),
                attempt_trace=list(call.get("attempt_trace") or []),
                api_error=observation.model_error,
                started_at_utc=started_at,
                completed_at_utc=completed_at,
            )
        )
    return records


def run_week2_live(
    project_root: str | Path,
    *,
    config_path: str | Path,
    caps: WholeRunCaps,
    catalog_verified_on: date,
    output_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    provider_factory: ProviderFactory = default_provider_factory,
    require_clean_git: bool = True,
    probe_sample_id: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> LiveWeek2Execution:
    """두 key와 모든 cap을 확인한 뒤 canonical 40건을 route별로 한 번 실행한다."""

    root = Path(project_root).resolve()
    config_file = _rooted(root, str(config_path))
    config = load_week2_live_config(config_file)
    verified_on = (
        date.fromisoformat(catalog_verified_on)
        if isinstance(catalog_verified_on, str)
        else catalog_verified_on
    )
    today = date.today()
    catalog_age = (today - verified_on).days
    if catalog_age < 0 or catalog_age > 7:
        raise Week2LiveError("catalog 확인 날짜는 오늘부터 7일 이내여야 합니다")
    for route in (config.baseline_route, config.candidate_route):
        pricing_age = (today - route.pricing_verified_on).days
        if pricing_age < 0 or pricing_age > 7:
            raise Week2LiveError(
                f"{route.provider_id} 가격 근거 날짜는 오늘부터 7일 이내여야 합니다"
            )
    selected_sample_count = 1 if probe_sample_id is not None else config.expected_sample_count
    validate_whole_run_caps_for_sample_count(
        config,
        caps,
        samples_per_route=selected_sample_count,
    )
    resolve_live_keys(config, environ)
    git = _collect_git_provenance(root)
    if require_clean_git and not git.clean:
        raise Week2LiveError("live_quality 실행은 변경사항이 없는 Git commit에서만 허용합니다")

    all_cases = build_cases(_rooted(root, config.paths.case_authoring))
    _validate_expected_cases(all_cases)
    input_manifest = build_live_input_manifest(root, config, all_cases)
    if probe_sample_id is None:
        cases = all_cases
    else:
        cases = [case for case in all_cases if case.sample_id == probe_sample_id]
        if len(cases) != 1:
            raise Week2LiveError(f"canonical probe sample_id를 찾을 수 없습니다: {probe_sample_id}")
    route_contracts = {
        "baseline": build_live_comparison_contract(root, config),
        "candidate": build_live_comparison_contract(root, config),
    }
    config_sha256 = sha256_file(config_file)
    whole_guard = _WholeRunGuard(caps, clock=clock)
    prepared_documents = _rooted(root, config.paths.prepared_documents)
    prompt_path = _rooted(root, config.paths.prompt)
    target = (
        _rooted(root, config.paths.output) if output_dir is None else _rooted(root, str(output_dir))
    )
    target.mkdir(parents=True, exist_ok=False)
    run_id = f"week02-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:12]}"
    run_started_at = datetime.now(UTC)
    (target / "input-manifest.json").write_text(
        input_manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        "artifact_schema_version": 3,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": run_started_at.isoformat(),
        "completed_at_utc": None,
        "evidence_kind": "live_quality",
        "fallback_enabled": False,
        "replay_enabled": False,
        "probe_sample_id": probe_sample_id,
        "target_sample_ids": [case.sample_id for case in cases],
        "caps": caps.model_dump(mode="json"),
        "catalog_verified_on": verified_on.isoformat(),
        "config_sha256": config_sha256,
        "input_manifest_sha256": input_manifest.sha256,
        "comparison_contract_sha256": route_contracts["baseline"].sha256,
        "routes": {
            role: {
                "provider_id": route.provider_id,
                "requested_model": route.model,
                "expected_actual_model": route.expected_actual_model,
                "api_key_env": route.api_key_env,
            }
            for role, route in (
                ("baseline", config.baseline_route),
                ("candidate", config.candidate_route),
            )
        },
    }
    atomic_write_json(target / "run-manifest.json", run_manifest)
    providers = {
        role: provider_factory(route)
        for role, route in (
            ("baseline", config.baseline_route),
            ("candidate", config.candidate_route),
        )
    }

    route_payloads: dict[
        str,
        tuple[
            list[ModelObservation],
            list[EvaluationResult],
            RouteProvenance,
        ],
    ] = {}
    for role, route in (
        ("baseline", config.baseline_route),
        ("candidate", config.candidate_route),
    ):
        started_at = datetime.now(UTC)
        attach_persistent_provider_budget(
            providers[role],
            target / f"{role}-budget.json",
        )
        attach_provider_response_journal(
            providers[role],
            target / f"{role}-provider-responses.jsonl",
        )
        provider = _BudgetedRouteProvider(
            route,
            providers[role],
            whole_guard,
            clock=clock,
        )
        observations_path = target / f"{role}-observations.jsonl"
        observations = run_cases(
            cases=cases,
            prepared_documents=prepared_documents,
            prompt_path=prompt_path,
            provider=provider,
            on_observation=lambda observation, path=observations_path: _append_jsonl_fsync(
                path,
                observation,
            ),
        )
        _validate_persisted_observations(observations_path, observations)
        results = score_observations(cases, observations)
        completed_at = datetime.now(UTC)
        live_records_file = f"{role}-live-records.jsonl"
        live_records = _build_live_sample_records(
            run_id=run_id,
            role=role,
            route=route,
            cases=cases,
            observations=observations,
            results=results,
            input_manifest=input_manifest,
            contract=route_contracts[role],
            config_sha256=config_sha256,
            git=git,
            started_at=started_at,
            completed_at=completed_at,
        )
        provenance = _route_provenance(
            role=role,
            route=route,
            observations=observations,
            results=results,
            git=git,
            catalog_verified_on=verified_on,
            config_sha256=config_sha256,
            contract=route_contracts[role],
            input_manifest=input_manifest,
            started_at=started_at,
            completed_at=completed_at,
            budget=provider.snapshot(),
            live_records_file=live_records_file,
        )
        route_payloads[role] = (observations, results, provenance)
        _write_jsonl(target / f"{role}-results.jsonl", results)
        _write_jsonl(target / live_records_file, live_records)
        (target / f"{role}-provenance.json").write_text(
            provenance.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    baseline_observations, baseline_results, baseline_provenance = route_payloads["baseline"]
    candidate_observations, candidate_results, candidate_provenance = route_payloads["candidate"]
    after_manifest = build_live_input_manifest(root, config, all_cases)
    extra_reasons: list[str] = []
    if after_manifest.sha256 != input_manifest.sha256:
        extra_reasons.append("실행 중 input manifest가 변경됨")
    for provenance in (baseline_provenance, candidate_provenance):
        if provenance.budget.violation:
            extra_reasons.append(
                f"{provenance.role} budget violation: {provenance.budget.violation}"
            )

    comparison = compare_routes(
        baseline_results,
        candidate_results,
        baseline_route=config.baseline_route.descriptor(),
        candidate_route=config.candidate_route.descriptor(),
        baseline_contract=route_contracts["baseline"],
        candidate_contract=route_contracts["candidate"],
    )
    comparison = enforce_live_comparison_requirements(
        comparison,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        baseline_route=config.baseline_route,
        candidate_route=config.candidate_route,
        extra_reasons=extra_reasons,
    )
    summary = LiveWeek2Summary(
        run_id=run_id,
        automated_status=comparison.automated_status,
        automated_reason=comparison.automated_reason,
        invalid_reasons=comparison.invalid_comparison_reasons,
        baseline_observation_count=len(baseline_observations),
        candidate_observation_count=len(candidate_observations),
        baseline_provider_errors=baseline_provenance.provider_error_count,
        candidate_provider_errors=candidate_provenance.provider_error_count,
        baseline_invalid_outputs=baseline_provenance.invalid_output_count,
        candidate_invalid_outputs=candidate_provenance.invalid_output_count,
        input_manifest_sha256=input_manifest.sha256,
        comparison_contract_sha256=route_contracts["baseline"].sha256,
    )

    (target / "comparison.json").write_text(
        comparison.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (target / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    run_manifest.update(
        {
            "status": summary.automated_status,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "baseline_observation_count": len(baseline_observations),
            "candidate_observation_count": len(candidate_observations),
            "invalid_reasons": summary.invalid_reasons,
        }
    )
    atomic_write_json(target / "run-manifest.json", run_manifest)
    return LiveWeek2Execution(
        summary=summary,
        comparison=comparison,
        baseline_provenance=baseline_provenance,
        candidate_provenance=candidate_provenance,
    )
