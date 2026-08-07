"""NVIDIA NIM VLM을 고유 run과 누적 예산 안에서 실행하고 즉시 평가한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from verifiable_ai_workflow.config import (
    LabSettings,
    load_project_env,
    load_settings,
    project_path,
)
from verifiable_ai_workflow.data.dataset import build_cases, load_cases
from verifiable_ai_workflow.evaluation.deepeval_runner import evaluate_results
from verifiable_ai_workflow.evaluation.scoring import SCORING_PROFILE, score_observations
from verifiable_ai_workflow.live_execution import (
    LiveBudget,
    LiveBudgetCaps,
    LiveBudgetState,
    LiveExecutionError,
    RunFileLock,
    atomic_write_json,
    require_canonical_project_file,
)
from verifiable_ai_workflow.preprocessing import load_document
from verifiable_ai_workflow.providers.litellm_provider import LiteLLMProvider
from verifiable_ai_workflow.schemas import EvaluationCase, ModelObservation
from verifiable_ai_workflow.workflow import run_cases
from verifiable_ai_workflow.workflow.inputs import build_page_input_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = "configs/nvidia-nim.yaml"
GEMMA_BASELINE_CONFIG = "configs/nvidia-nim-gemma4-baseline.yaml"
GEMMA_IMPROVED_CONFIG = "configs/nvidia-nim-gemma4.yaml"
APPROVED_CONFIGS = (
    CANONICAL_CONFIG,
    GEMMA_BASELINE_CONFIG,
    GEMMA_IMPROVED_CONFIG,
)
CANONICAL_CASE_AUTHORING = "data/cases/week-01-aihub.yaml"
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_TASK_KEY_ENV = "NVIDIA_NIM_API_KEY"
APPROVED_MODELS_BY_CONFIG = {
    CANONICAL_CONFIG: (
        "nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    ),
    GEMMA_BASELINE_CONFIG: (
        "nvidia_nim/google/gemma-4-31b-it",
        "google/gemma-4-31b-it",
    ),
    GEMMA_IMPROVED_CONFIG: (
        "nvidia_nim/google/gemma-4-31b-it",
        "google/gemma-4-31b-it",
    ),
}
PROBE_SAMPLE_ID = "aihub-report-r01"
RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
PROVENANCE_COMPONENTS = (
    "scripts/run_nvidia_nim.py",
    "src/verifiable_ai_workflow/live_execution.py",
    "src/verifiable_ai_workflow/providers/litellm_provider.py",
    "src/verifiable_ai_workflow/workflow/runner.py",
    "src/verifiable_ai_workflow/workflow/inputs.py",
)


def _require_approved_config(supplied_path: str | Path) -> Path:
    """검토한 NVIDIA NIM 설정만 실제 API 실행에 허용한다."""

    root = PROJECT_ROOT.resolve()
    supplied = Path(supplied_path)
    if not supplied.is_absolute():
        supplied = root / supplied
    supplied_resolved = supplied.resolve()
    for canonical in APPROVED_CONFIGS:
        if supplied_resolved == root / canonical:
            return require_canonical_project_file(root, supplied, canonical)
    approved = ", ".join(APPROVED_CONFIGS)
    raise LiveExecutionError(f"승인된 NVIDIA NIM 설정만 사용할 수 있습니다: {approved}")


def _classify_run_status(
    *,
    probe_only: bool,
    blocked: bool,
    complete: bool,
    provider_error_count: int,
    model_drift_count: int,
) -> tuple[str, str]:
    """개별 관찰 결과와 품질 주장에 쓸 run-level 상태를 분리한다."""

    if blocked:
        observed_status = "blocked"
    elif not complete:
        observed_status = "partial"
    elif provider_error_count or model_drift_count:
        observed_status = "inconclusive"
    else:
        observed_status = "complete"
    run_status = "inconclusive" if probe_only else observed_status
    return run_status, observed_status


def _require_approved_case_copy(
    *,
    canonical_cases: list[EvaluationCase],
    local_cases: list[EvaluationCase],
) -> list[EvaluationCase]:
    """Git 제외 생성물을 tracked Week 1 40건과 exact 비교한다."""

    if len(canonical_cases) != 40:
        raise ValueError("canonical Week 1 authoring data는 정확히 40건이어야 합니다")
    if any(case.split == "sealed_test" for case in canonical_cases + local_cases):
        raise ValueError("Week 1 live 실행에는 sealed_test 입력을 사용할 수 없습니다")
    canonical_payload = [case.model_dump(mode="json") for case in canonical_cases]
    local_payload = [case.model_dump(mode="json") for case in local_cases]
    if local_payload != canonical_payload:
        raise ValueError(
            "local cases.jsonl은 tracked Week 1 authoring data에서 만든 exact 40건과 같아야 합니다"
        )
    return canonical_cases


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("양수여야 합니다")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("0 이상이어야 합니다")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("양의 유한수여야 합니다")
    return parsed


def _require_approved_provider(settings: LabSettings, config_path: Path) -> None:
    config_name = config_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    requested_model, expected_actual_model = APPROVED_MODELS_BY_CONFIG[config_name]
    actual = {
        "model": settings.provider.model,
        "expected_actual_model": settings.provider.expected_actual_model,
        "api_base": settings.provider.api_base,
        "api_key_env": settings.provider.api_key_env,
    }
    expected = {
        "model": requested_model,
        "expected_actual_model": expected_actual_model,
        "api_base": NVIDIA_API_BASE,
        "api_key_env": NVIDIA_TASK_KEY_ENV,
    }
    if actual != expected:
        raise ValueError(
            "live 호출은 설정별로 승인된 NVIDIA endpoint·key 환경 변수·model만 허용합니다"
        )


def _with_probe_prompt(settings: LabSettings, supplied_path: str | Path) -> LabSettings:
    prompt_path = project_path(PROJECT_ROOT, str(supplied_path))
    local_data = (PROJECT_ROOT / "local-data").resolve()
    if not prompt_path.is_relative_to(local_data) or not prompt_path.is_file():
        raise ValueError("학습자 prompt는 local-data 아래의 기존 파일이어야 합니다")
    return settings.model_copy(
        update={
            "paths": settings.paths.model_copy(
                update={"prompt": prompt_path.relative_to(PROJECT_ROOT).as_posix()}
            )
        }
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_state() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, not bool(dirty.strip())


def _build_input_manifest(
    cases: list[EvaluationCase],
    prepared_documents: Path,
    max_images_per_prompt: int | None,
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    for document_id in sorted({case.document_id for case in cases}):
        document, manifest_path = load_document(prepared_documents, document_id)
        pages = []
        for page in document.pages:
            model_image_path = manifest_path.parent / page.model_image_path
            pages.append(
                {
                    "page_number": page.page_number,
                    "model_image_path": page.model_image_path,
                    "model_image_bytes": model_image_path.stat().st_size,
                    "model_image_sha256": _sha256_file(model_image_path),
                }
            )
        document_record = {
            "document_id": document.document_id,
            "source_file": document.source_file,
            "source_sha256": document.source_sha256,
            "prepared_manifest_sha256": _sha256_file(manifest_path),
            "total_pages": document.total_pages,
            "render_dpi": document.render_dpi,
            "pages": pages,
        }
        if max_images_per_prompt is not None:
            document_record["model_input_images"] = build_page_input_manifest(
                document,
                manifest_path,
                max_images_per_prompt,
            )
        documents.append(document_record)
    return {
        "artifact_schema_version": 2,
        "input_modality": "page_images_only",
        "scoring_profile": SCORING_PROFILE,
        "sample_ids": sorted(case.sample_id for case in cases),
        "documents": documents,
    }


def _build_provenance(
    *,
    settings: LabSettings,
    config_path: Path,
    cases: list[EvaluationCase],
    input_manifest: dict[str, Any],
    catalog_verified_on: date,
    require_clean_git: bool,
) -> dict[str, Any]:
    git_sha, git_clean = _git_state()
    if require_clean_git and not git_clean:
        raise RuntimeError("전체 품질 실행은 변경사항이 없는 Git commit에서만 허용합니다")

    component_hashes = {
        relative: _sha256_file(PROJECT_ROOT / relative) for relative in PROVENANCE_COMPONENTS
    }
    cases_path = project_path(PROJECT_ROOT, settings.paths.cases)
    source_rows = sorted(
        {
            (
                case.source.title,
                case.source.license,
                case.source.revision,
            )
            for case in cases
        }
    )
    return {
        "git_sha": git_sha,
        "git_clean": git_clean,
        "config_sha256": _sha256_file(config_path),
        "dataset_sha256": _sha256_file(cases_path),
        "input_manifest_content_sha256": _canonical_sha256(input_manifest),
        "lockfile_sha256": _sha256_file(PROJECT_ROOT / "uv.lock"),
        "prompt_sha256": _sha256_file(project_path(PROJECT_ROOT, settings.paths.prompt)),
        "schema_sha256": _sha256_file(
            PROJECT_ROOT / "src/verifiable_ai_workflow/schemas/models.py"
        ),
        "scorer_sha256": _sha256_file(
            PROJECT_ROOT / "src/verifiable_ai_workflow/evaluation/scoring.py"
        ),
        "workflow_sha256": _canonical_sha256(component_hashes),
        "workflow_component_sha256": component_hashes,
        "catalog_verified_on": catalog_verified_on.isoformat(),
        "billing_basis": settings.provider.billing_basis,
        "pricing_source_url": settings.provider.pricing_source_url,
        "pricing_verified_on": (
            settings.provider.pricing_verified_on.isoformat()
            if settings.provider.pricing_verified_on is not None
            else None
        ),
        "sources": [
            {"title": title, "license": license_name, "revision": revision}
            for title, license_name, revision in source_rows
        ],
    }


def _new_run_id(prefix: str = "week01") -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{secrets.token_hex(4)}"


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id는 영문·숫자로 시작하는 안전한 파일 이름이어야 합니다")
    return run_id


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _load_observations(path: Path) -> list[ModelObservation]:
    if not path.is_file():
        return []
    observations = [
        ModelObservation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample_ids = [observation.sample_id for observation in observations]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("live run observation에 중복 sample_id가 있습니다")
    return observations


def _validate_cli_caps(
    args: argparse.Namespace,
    settings: LabSettings,
    *,
    target_count: int,
) -> LiveBudgetCaps:
    config_limits = settings.limits
    comparisons = (
        ("max_requests", args.max_requests, config_limits.max_requests),
        ("max_input_tokens", args.max_input_tokens, config_limits.max_input_tokens),
        ("max_output_tokens", args.max_output_tokens, config_limits.max_output_tokens),
        ("max_cost_usd", args.max_cost_usd, config_limits.max_cost_usd),
        ("max_wall_seconds", args.max_wall_seconds, config_limits.max_wall_seconds),
        ("max_retries", args.max_retries, config_limits.max_retries),
    )
    exceeded = [
        f"{name}={actual} > config ceiling {ceiling}"
        for name, actual, ceiling in comparisons
        if actual > ceiling
    ]
    if exceeded:
        raise ValueError("CLI budget이 config ceiling을 넘습니다: " + "; ".join(exceeded))
    if args.max_requests != target_count:
        raise ValueError(f"max_requests는 target {target_count}건과 정확히 같아야 합니다")
    request_input_ceiling = config_limits.request_input_token_ceiling
    request_output_ceiling = config_limits.request_output_token_ceiling
    if request_input_ceiling is None or request_output_ceiling is None:
        raise ValueError("config에 request별 input/output token ceiling이 필요합니다")
    if request_input_ceiling > args.max_input_tokens:
        raise ValueError("전체 input token cap이 request input ceiling보다 작습니다")
    if request_output_ceiling > args.max_output_tokens:
        raise ValueError("전체 output token cap이 request output ceiling보다 작습니다")
    required_attempts = target_count * (args.max_retries + 1)
    required_input_tokens = required_attempts * request_input_ceiling
    required_output_tokens = required_attempts * request_output_ceiling
    if args.max_input_tokens < required_input_tokens:
        raise ValueError(
            f"전체 input token cap은 target 예약량 {required_input_tokens} 이상이어야 합니다"
        )
    if args.max_output_tokens < required_output_tokens:
        raise ValueError(
            f"전체 output token cap은 target 예약량 {required_output_tokens} 이상이어야 합니다"
        )
    request_timeout = config_limits.request_timeout_seconds
    if request_timeout is None:
        raise ValueError("config에 request timeout이 필요합니다")
    pacing_wait = max(0, required_attempts - 1) * (60.0 / config_limits.requests_per_minute)
    required_wall = required_attempts * request_timeout + pacing_wait
    if args.max_wall_seconds < required_wall:
        raise ValueError(
            f"전체 wall cap은 timeout·rate 대기 예약량 {required_wall:g} 이상이어야 합니다"
        )
    approved_profiles = {
        1: {
            "max_input_tokens": 20_000,
            "max_output_tokens": 500,
            "max_cost_usd": 0.01,
            "max_wall_seconds": 120,
        },
        40: {
            "max_input_tokens": 800_000,
            "max_output_tokens": 20_000,
            "max_cost_usd": 0.01,
            "max_wall_seconds": 7_200,
        },
    }
    approved = approved_profiles.get(target_count)
    if approved is None:
        raise ValueError("NVIDIA live target은 canonical probe 1건 또는 full 40건이어야 합니다")
    mismatches = [
        f"{name}={getattr(args, name)} != approved={value}"
        for name, value in approved.items()
        if not math.isclose(getattr(args, name), value, rel_tol=0.0, abs_tol=1e-12)
    ]
    if mismatches:
        raise ValueError("NVIDIA live 승인 cap과 다릅니다: " + "; ".join(mismatches))
    return LiveBudgetCaps(
        max_requests=args.max_requests,
        max_attempts=args.max_requests * (args.max_retries + 1),
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_cost_usd=args.max_cost_usd,
        max_wall_seconds=args.max_wall_seconds,
    )


def _immutable_run_contract(
    *,
    run_id: str,
    trial_id: str,
    settings: LabSettings,
    caps: LiveBudgetCaps,
    target_cases: list[EvaluationCase],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if not settings.provider.expected_actual_model:
        raise ValueError("live provider에 expected_actual_model이 필요합니다")
    return {
        "run_id": run_id,
        "trial_id": trial_id,
        "target_sample_ids": sorted(case.sample_id for case in target_cases),
        "provider": {
            "adapter": "litellm",
            "requested_model": settings.provider.model,
            "expected_actual_model": settings.provider.expected_actual_model,
            "api_base": settings.provider.api_base,
            "structured_output": settings.provider.structured_output,
            "temperature": settings.provider.temperature,
            "top_p": settings.provider.top_p,
            "seed": settings.provider.seed,
            "thinking_mode": settings.provider.thinking_mode,
            "thinking_parameter": settings.provider.thinking_parameter,
            "max_images_per_prompt": settings.provider.max_images_per_prompt,
            "api_key_env": settings.provider.api_key_env,
            "billing_basis": settings.provider.billing_basis,
            "pricing_source_url": settings.provider.pricing_source_url,
            "pricing_verified_on": (
                settings.provider.pricing_verified_on.isoformat()
                if settings.provider.pricing_verified_on is not None
                else None
            ),
        },
        "evaluation_mode": "benchmark",
        "evidence_kind": "live_quality",
        "fallback_enabled": False,
        "replay_enabled": False,
        "caps": caps.model_dump(mode="json"),
        "provenance": provenance,
    }


def _write_records(
    *,
    path: Path,
    run_contract: dict[str, Any],
    cases: list[EvaluationCase],
    observations: list[ModelObservation],
    results: list[Any],
) -> None:
    case_by_id = {case.sample_id: case for case in cases}
    observation_by_id = {observation.sample_id: observation for observation in observations}
    temporary = path.with_name(f".{path.name}.tmp-{secrets.token_hex(8)}")
    with temporary.open("x", encoding="utf-8") as handle:
        for result in results:
            case = case_by_id[result.sample_id]
            observation = observation_by_id[result.sample_id]
            call = observation.model_call or {}
            record = {
                "artifact_schema_version": 3,
                "run_id": run_contract["run_id"],
                "trial_id": run_contract["trial_id"],
                "sample_id": case.sample_id,
                "family_id": case.family_id,
                "split": case.split,
                "risk_level": case.risk_level,
                "source": case.source.model_dump(mode="json"),
                "observed_at": (
                    call.get("attempt_trace", [{}])[-1].get("completed_at")
                    if call.get("attempt_trace")
                    else None
                ),
                "provider": run_contract["provider"],
                "evaluation_mode": "benchmark",
                "evidence_kind": "live_quality",
                "fallback_enabled": False,
                "replay_enabled": False,
                "provenance": run_contract["provenance"],
                "raw_response": observation.raw_output,
                "parsed_response": (
                    result.output.model_dump(mode="json") if result.output else None
                ),
                "provider_error": observation.model_error,
                "model_call": observation.model_call,
                "scores": result.scores,
                "reasons": result.reasons,
                "status": result.status,
            }
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NVIDIA NIM Week 1 bounded live batch")
    parser.add_argument("--config", default=CANONICAL_CONFIG)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-requests", type=_positive_int, required=True)
    parser.add_argument("--max-input-tokens", type=_positive_int, required=True)
    parser.add_argument("--max-output-tokens", type=_positive_int, required=True)
    parser.add_argument("--max-cost-usd", type=_positive_float, required=True)
    parser.add_argument("--max-wall-seconds", type=_positive_float, required=True)
    parser.add_argument("--max-retries", type=_nonnegative_int, required=True)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--sample-id", choices=(PROBE_SAMPLE_ID,))
    parser.add_argument(
        "--prompt",
        help="한 사례 probe에서만 사용하는 local-data 아래의 학습자 prompt",
    )
    parser.add_argument("--trial-id", default="trial-01")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--catalog-verified-on",
        type=date.fromisoformat,
        required=True,
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if not args.live:
        parser.error("실제 API 호출에는 --live가 필요합니다")
    if args.resume and not args.run_id:
        parser.error("--resume에는 기존 --run-id가 필요합니다")
    if not args.resume and args.run_id:
        parser.error("새 run_id는 자동 생성됩니다. --run-id는 --resume에만 사용합니다")
    if args.resume and args.sample_id:
        parser.error("--resume에서는 최초 run의 target을 변경할 수 없습니다")
    if args.prompt and (args.resume or not args.sample_id):
        parser.error("--prompt는 새 한 사례 probe에서만 사용할 수 있습니다")
    _validate_run_id(args.trial_id)

    load_project_env(PROJECT_ROOT)
    config_path = _require_approved_config(args.config)
    settings = load_settings(config_path)
    if args.prompt:
        settings = _with_probe_prompt(settings, args.prompt)
    if settings.provider.kind != "litellm":
        raise ValueError("NVIDIA NIM 설정의 provider.kind는 litellm이어야 합니다")
    _require_approved_provider(settings, config_path)
    if not settings.provider.api_key_env or not settings.provider.expected_actual_model:
        raise ValueError("NVIDIA NIM 설정에 key env와 expected actual model이 필요합니다")
    catalog_age = (date.today() - args.catalog_verified_on).days
    if catalog_age < 0 or catalog_age > 7:
        raise ValueError("--catalog-verified-on은 오늘부터 7일 이내여야 합니다")
    if settings.provider.pricing_verified_on is None:
        raise ValueError("NVIDIA NIM 가격 근거 날짜가 없습니다")
    pricing_age = (date.today() - settings.provider.pricing_verified_on).days
    if pricing_age < 0 or pricing_age > 7:
        raise ValueError("NVIDIA NIM 가격 근거 날짜는 오늘부터 7일 이내여야 합니다")

    case_authoring_path = require_canonical_project_file(
        PROJECT_ROOT,
        settings.paths.case_authoring,
        CANONICAL_CASE_AUTHORING,
    )
    all_cases = _require_approved_case_copy(
        canonical_cases=build_cases(case_authoring_path),
        local_cases=load_cases(project_path(PROJECT_ROOT, settings.paths.cases)),
    )
    case_by_id = {case.sample_id: case for case in all_cases}
    output_root = project_path(PROJECT_ROOT, settings.paths.output) / "runs"
    if args.resume:
        run_id = _validate_run_id(args.run_id)
        run_dir = output_root / run_id
        manifest_path = run_dir / "run-manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"resume run manifest를 찾을 수 없습니다: {run_id}")
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target_ids = prior_manifest["contract"]["target_sample_ids"]
        if any(sample_id not in case_by_id for sample_id in target_ids):
            raise ValueError("현재 dataset에 최초 run의 target sample이 없습니다")
        target_cases = [case_by_id[sample_id] for sample_id in target_ids]
        trial_id = prior_manifest["contract"]["trial_id"]
        if args.trial_id != "trial-01" and args.trial_id != trial_id:
            raise ValueError("resume에서 trial_id를 변경할 수 없습니다")
    else:
        prefix = {
            Path(GEMMA_BASELINE_CONFIG).name: "week02-gemma-baseline",
            Path(GEMMA_IMPROVED_CONFIG).name: "week02-gemma-improved",
        }.get(config_path.name, "week01")
        run_id = _new_run_id(prefix)
        target_cases = all_cases
        if args.sample_id:
            target_cases = [case for case in all_cases if case.sample_id == args.sample_id]
            if len(target_cases) != 1:
                raise ValueError(f"sample_id를 찾을 수 없습니다: {args.sample_id}")
        trial_id = args.trial_id
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = run_dir / "run-manifest.json"
        prior_manifest = None

    caps = _validate_cli_caps(args, settings, target_count=len(target_cases))
    prepared_documents = project_path(PROJECT_ROOT, settings.paths.prepared_documents)
    input_manifest = _build_input_manifest(
        target_cases,
        prepared_documents,
        settings.provider.max_images_per_prompt,
    )
    provenance = _build_provenance(
        settings=settings,
        config_path=config_path,
        cases=target_cases,
        input_manifest=input_manifest,
        catalog_verified_on=args.catalog_verified_on,
        require_clean_git=len(target_cases) > 1,
    )
    run_contract = _immutable_run_contract(
        run_id=run_id,
        trial_id=trial_id,
        settings=settings,
        caps=caps,
        target_cases=target_cases,
        provenance=provenance,
    )

    with RunFileLock(run_dir / "run.lock"):
        input_manifest_path = run_dir / "input-manifest.json"
        budget_path = run_dir / "budget.json"
        observations_path = run_dir / "observations.jsonl"
        if args.resume:
            if prior_manifest is None or prior_manifest.get("contract") != run_contract:
                raise ValueError("현재 Git·dataset·입력·config·budget 설정이 최초 run과 다릅니다")
            if not budget_path.is_file() or not input_manifest_path.is_file():
                raise ValueError("resume에 필요한 budget 또는 input manifest가 없습니다")
            if _sha256_file(input_manifest_path) != prior_manifest["input_manifest_sha256"]:
                raise ValueError("저장된 input manifest exact-byte hash가 다릅니다")
            state = LiveBudgetState.model_validate_json(budget_path.read_text(encoding="utf-8"))
            budget = LiveBudget(caps, state=state)
            budget.set_on_change(
                lambda value: atomic_write_json(budget_path, value.persisted_dict())
            )
            budget.recover_interrupted_attempts()
            started_at = prior_manifest["started_at"]
        else:
            atomic_write_json(input_manifest_path, input_manifest)
            budget = LiveBudget(caps)
            budget.set_on_change(
                lambda value: atomic_write_json(budget_path, value.persisted_dict())
            )
            atomic_write_json(budget_path, budget.state.persisted_dict())
            started_at = datetime.now(UTC).isoformat()
            prior_manifest = {
                "artifact_schema_version": 3,
                "contract": run_contract,
                "input_manifest_sha256": _sha256_file(input_manifest_path),
                "started_at": started_at,
                "updated_at": started_at,
                "completed_at": None,
                "status": "running",
                "live_call_performed": False,
                "budget": budget.summary(),
            }
            atomic_write_json(manifest_path, prior_manifest)

        observations = _load_observations(observations_path)
        completed_ids = {observation.sample_id for observation in observations}
        if not completed_ids <= set(run_contract["target_sample_ids"]):
            raise ValueError("run target 밖의 observation이 섞여 있습니다")
        pending = [case for case in target_cases if case.sample_id not in completed_ids]
        if args.limit is not None:
            pending = pending[: args.limit]

        provider = LiteLLMProvider(
            model=settings.provider.model,
            expected_actual_model=settings.provider.expected_actual_model,
            api_key_env=settings.provider.api_key_env,
            api_base=settings.provider.api_base,
            structured_output=settings.provider.structured_output,
            max_requests=caps.max_requests,
            max_attempts=caps.max_attempts,
            requests_per_minute=settings.limits.requests_per_minute,
            max_retries=args.max_retries,
            retry_initial_seconds=settings.limits.retry_initial_seconds,
            max_cost_usd=caps.max_cost_usd,
            max_input_tokens=caps.max_input_tokens,
            max_output_tokens=caps.max_output_tokens,
            max_wall_seconds=caps.max_wall_seconds,
            request_input_token_ceiling=settings.limits.request_input_token_ceiling,
            request_output_token_ceiling=settings.limits.request_output_token_ceiling,
            request_timeout_seconds=settings.limits.request_timeout_seconds,
            input_cost_per_token_usd=settings.provider.input_cost_per_token_usd,
            output_cost_per_token_usd=settings.provider.output_cost_per_token_usd,
            temperature=settings.provider.temperature,
            top_p=settings.provider.top_p,
            seed=settings.provider.seed,
            thinking_mode=settings.provider.thinking_mode,
            thinking_parameter=settings.provider.thinking_parameter,
            max_images_per_prompt=settings.provider.max_images_per_prompt,
            budget=budget,
            resume_last_attempt_started_at=(
                max(
                    (attempt.started_at for attempt in budget.state.attempts),
                    default=None,
                )
                if args.resume
                else None
            ),
            on_response_received=lambda record: _append_jsonl(
                run_dir / "provider-responses.jsonl",
                record,
            ),
        )
        blocked = False
        for index, case in enumerate(pending, start=1):
            observation = run_cases(
                cases=[case],
                prepared_documents=prepared_documents,
                prompt_path=project_path(PROJECT_ROOT, settings.paths.prompt),
                provider=provider,
            )[0]
            call_status = (
                observation.model_call.get("provider_status") if observation.model_call else None
            )
            observation = observation.model_copy(
                update={
                    "provider_status": (
                        call_status
                        if call_status in {"success", "provider_error", "blocked"}
                        else "provider_error"
                    )
                }
            )
            if call_status == "blocked":
                _append_jsonl(
                    run_dir / "blocked-events.jsonl",
                    {
                        "run_id": run_id,
                        "sample_id": case.sample_id,
                        "occurred_at": datetime.now(UTC).isoformat(),
                        "model_call": observation.model_call,
                        "error": observation.model_error,
                    },
                )
                blocked = True
                break
            _append_jsonl(
                observations_path,
                observation.model_dump(mode="json"),
            )
            result = score_observations([case], [observation])[0]
            print(
                f"[{len(completed_ids) + index}/{len(target_cases)}] "
                f"{case.sample_id}: {result.status}"
            )
            prior_manifest.update(
                {
                    "updated_at": datetime.now(UTC).isoformat(),
                    "status": "running",
                    "live_call_performed": budget.attempt_count > 0,
                    "budget": budget.summary(),
                }
            )
            atomic_write_json(manifest_path, prior_manifest)

        all_observations = _load_observations(observations_path)
        evaluated_cases = [case_by_id[observation.sample_id] for observation in all_observations]
        results = score_observations(evaluated_cases, all_observations)
        (run_dir / "results.jsonl").write_text(
            "".join(result.model_dump_json() + "\n" for result in results),
            encoding="utf-8",
        )
        if results:
            evaluate_results(results, evaluated_cases, run_dir / "deepeval")
        _write_records(
            path=run_dir / "records.jsonl",
            run_contract=run_contract,
            cases=evaluated_cases,
            observations=all_observations,
            results=results,
        )

        score_names = tuple(results[0].scores) if results else ()
        actual_models = sorted(
            {
                str(observation.model_call["actual_model"])
                for observation in all_observations
                if observation.model_call and observation.model_call.get("actual_model") is not None
            }
        )
        model_drift_count = sum(
            bool(observation.model_call)
            and not observation.model_call.get("actual_model_matches_expected", False)
            for observation in all_observations
            if observation.model_error is None
        )
        provider_error_count = sum(
            observation.model_error is not None for observation in all_observations
        )
        complete = len(all_observations) == len(target_cases)
        probe_only = len(target_cases) == 1
        status, observed_status = _classify_run_status(
            probe_only=probe_only,
            blocked=blocked,
            complete=complete,
            provider_error_count=provider_error_count,
            model_drift_count=model_drift_count,
        )
        summary = {
            "artifact_schema_version": 3,
            "run_id": run_id,
            "trial_id": trial_id,
            "status": status,
            "observed_status": observed_status,
            "probe_only": probe_only,
            "probe_reason": (
                "한 사례의 연결·raw response·actual model·usage 확인용이며 "
                "품질 완료 근거가 아닙니다"
                if probe_only
                else None
            ),
            "record_count": len(results),
            "target_count": len(target_cases),
            "status_counts": dict(Counter(result.status for result in results)),
            "score_averages": {
                name: round(
                    sum(result.scores[name] for result in results) / len(results),
                    4,
                )
                for name in score_names
            },
            "evidence_kind": "live_quality",
            "evaluation_mode": "benchmark",
            "fallback_enabled": False,
            "replay_enabled": False,
            "live_call_performed": budget.attempt_count > 0,
            "judge_status": "not_requested",
            "requested_model": settings.provider.model,
            "expected_actual_model": settings.provider.expected_actual_model,
            "actual_models": actual_models,
            "model_drift_count": model_drift_count,
            "provider_error_count": provider_error_count,
            "budget": budget.summary(),
            "provenance": provenance,
            "run_manifest": "run-manifest.json",
        }
        atomic_write_json(run_dir / "summary.json", summary)
        completed_at = datetime.now(UTC).isoformat() if complete else None
        prior_manifest.update(
            {
                "updated_at": datetime.now(UTC).isoformat(),
                "completed_at": completed_at,
                "status": status,
                "observed_status": observed_status,
                "probe_only": probe_only,
                "live_call_performed": budget.attempt_count > 0,
                "budget": budget.summary(),
                "record_count": len(results),
                "target_count": len(target_cases),
                "provider_error_count": provider_error_count,
                "model_drift_count": model_drift_count,
            }
        )
        atomic_write_json(manifest_path, prior_manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"run directory: {run_dir}")
        return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
