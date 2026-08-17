"""NVIDIA NIM Gemma로 OpenCQA baseline·improved 후보 30쌍을 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from verifiable_ai_workflow.config import load_project_env, load_settings
from verifiable_ai_workflow.live_execution import LiveBudgetCaps, atomic_write_json
from verifiable_ai_workflow.open_cqa_candidates import (
    CandidatePairDraft,
    bind_candidate_set_sha256,
    generate_candidate_pairs,
    load_open_cqa_cases,
    task_input_sha256,
)
from verifiable_ai_workflow.providers.litellm_provider import LiteLLMProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = PROJECT_ROOT / "local-data/opencqa/week-03-cases.jsonl"
SELECTION = PROJECT_ROOT / "data/opencqa/week-03-selection.yaml"
BASELINE_PROMPT = PROJECT_ROOT / "prompts/open-cqa-answer-baseline.md"
IMPROVED_PROMPT = PROJECT_ROOT / "prompts/open-cqa-answer-improved.md"
PROVIDER_CONFIG = PROJECT_ROOT / "configs/week-03-candidates.yaml"
IMAGE_ROOT = (PROJECT_ROOT / "local-data/opencqa/images").resolve()
APPROVED_PROVIDER = {
    "kind": "litellm",
    "model": "nvidia_nim/google/gemma-4-31b-it",
    "expected_actual_model": "google/gemma-4-31b-it",
    "api_base": "https://integrate.api.nvidia.com/v1",
    "api_key_env": "NVIDIA_NIM_API_KEY",
    "structured_output": "json_schema",
    "billing_basis": "developer_program_free_endpoint",
    "pricing_source_url": "https://docs.api.nvidia.com/nim/docs/product",
    "input_cost_per_token_usd": 0.0,
    "output_cost_per_token_usd": 0.0,
}
APPROVED_REQUEST_LIMITS = {
    "max_requests": 60,
    "requests_per_minute": 20,
    "max_retries": 0,
    "request_input_token_ceiling": 20_000,
    "request_output_token_ceiling": 500,
    "request_timeout_seconds": 120.0,
    "max_cost_usd": 0.01,
    "max_input_tokens": 1_200_000,
    "max_output_tokens": 30_000,
    "max_wall_seconds": 7_200.0,
}
APPROVED_CAPS = {
    1: LiveBudgetCaps(
        max_requests=2,
        max_attempts=2,
        max_input_tokens=40_000,
        max_output_tokens=1_000,
        max_cost_usd=0.01,
        max_wall_seconds=300,
    ),
    30: LiveBudgetCaps(
        max_requests=60,
        max_attempts=60,
        max_input_tokens=1_200_000,
        max_output_tokens=30_000,
        max_cost_usd=0.01,
        max_wall_seconds=7_200,
    ),
}


def _attempt_identity(record: dict[str, Any]) -> tuple[Any, Any]:
    return record.get("request_number"), record.get("attempt_number")


PROVENANCE_COMPONENTS = (
    "scripts/run_open_cqa_candidates.py",
    "src/verifiable_ai_workflow/open_cqa_candidates.py",
    "src/verifiable_ai_workflow/live_execution.py",
    "src/verifiable_ai_workflow/providers/litellm_provider.py",
    "src/verifiable_ai_workflow/evaluation/scoring.py",
    "src/verifiable_ai_workflow/schemas/models.py",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return sha, not dirty


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _select_cases(all_cases, pair_limit: int, pair_number: int):
    if pair_limit not in {1, 30} or pair_limit > len(all_cases):
        raise SystemExit("--pair-limit는 승인된 1 또는 30이어야 합니다")
    if pair_limit == 30:
        if pair_number != 1:
            raise SystemExit("30쌍 실행에서는 --pair-number 1만 사용할 수 있습니다")
        return all_cases
    if not 1 <= pair_number <= len(all_cases):
        raise SystemExit(f"--pair-number는 1부터 {len(all_cases)}까지입니다")
    return [all_cases[pair_number - 1]]


def _require_canonical_cases(cases) -> None:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8"))
    expected = [
        (f"opencqa-{selection['source_split']}-{sample_id}", sample_id, course_split)
        for course_split in ("development", "validation", "test")
        for sample_id in selection["course_splits"][course_split]
    ]
    actual = [(case.pair_id, case.sample_id, case.course_split) for case in cases]
    if actual != expected:
        raise SystemExit("OpenCQA case ID·순서·course split이 승인된 선택과 다릅니다")
    if any(
        case.source_split != selection["source_split"]
        or case.source_revision != selection["revision"]
        or case.source_license != selection["license"]
        for case in cases
    ):
        raise SystemExit("OpenCQA case의 split·revision·license가 승인된 선택과 다릅니다")


def _require_approved_provider(settings) -> None:
    actual_provider = {
        field: getattr(settings.provider, field) for field in APPROVED_PROVIDER
    }
    if actual_provider != APPROVED_PROVIDER:
        raise SystemExit("승인된 NVIDIA NIM Gemma endpoint·key·model만 사용합니다")
    actual_limits = {
        field: getattr(settings.limits, field) for field in APPROVED_REQUEST_LIMITS
    }
    if actual_limits != APPROVED_REQUEST_LIMITS:
        raise SystemExit("승인된 Week 3 전체·요청당 상한만 사용합니다")


def _validate_images(cases, *, max_bytes: int, max_width: int) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for case in cases:
        path = (PROJECT_ROOT / case.image_path).resolve()
        if not path.is_relative_to(IMAGE_ROOT) or not path.is_file():
            raise SystemExit(f"OpenCQA 이미지 경로를 확인하세요: {case.pair_id}")
        if _sha256(path) != case.image_sha256:
            raise SystemExit(f"OpenCQA 이미지 SHA-256이 다릅니다: {case.pair_id}")
        if path.suffix.casefold() not in {".jpg", ".jpeg"} or path.stat().st_size > max_bytes:
            raise SystemExit(
                f"OpenCQA 이미지는 {max_bytes} bytes 이하 JPEG여야 합니다: "
                f"{case.pair_id}"
            )
        try:
            with Image.open(path) as image:
                if image.format != "JPEG" or image.width > max_width:
                    raise SystemExit(
                        f"OpenCQA 이미지는 너비 {max_width}px 이하 JPEG여야 "
                        f"합니다: {case.pair_id}"
                    )
                image.verify()
        except OSError as exc:
            raise SystemExit(f"OpenCQA JPEG을 읽을 수 없습니다: {case.pair_id}") from exc
        images[case.pair_id] = path
    return images


def _validate_dates(catalog_verified_on: date, pricing_verified_on: date) -> None:
    for flag, checked_on in (
        ("--catalog-verified-on", catalog_verified_on),
        ("--pricing-verified-on", pricing_verified_on),
    ):
        if checked_on != date.today():
            raise SystemExit(f"{flag}은 실행 당일 날짜여야 합니다")


def _requested_caps(args) -> LiveBudgetCaps:
    return LiveBudgetCaps(
        max_requests=args.max_requests,
        max_attempts=args.max_attempts,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_cost_usd=args.max_cost_usd,
        max_wall_seconds=args.max_wall_seconds,
    )


def _write_results(path: Path, pairs) -> None:
    path.write_text(
        "".join(pair.model_dump_json() + "\n" for pair in pairs),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-task", action="store_true")
    parser.add_argument("--pair-limit", type=int, required=True)
    parser.add_argument("--pair-number", type=int, default=1)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, required=True)
    parser.add_argument("--max-retries", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--catalog-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument("--pricing-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.live_task:
        raise SystemExit("실제 task model 호출에는 --live-task가 필요합니다")
    if args.max_retries != 0:
        raise SystemExit("OpenCQA 후보 생성은 retry 0회만 허용합니다")
    _validate_dates(args.catalog_verified_on, args.pricing_verified_on)
    for path in (CASES, SELECTION, BASELINE_PROMPT, IMPROVED_PROMPT, PROVIDER_CONFIG):
        if not path.is_file():
            raise SystemExit(f"준비 파일이 없습니다: {path}")

    all_cases = load_open_cqa_cases(CASES)
    if len(all_cases) != 30:
        raise SystemExit("OpenCQA 준비 case는 정확히 30개여야 합니다")
    _require_canonical_cases(all_cases)
    cases = _select_cases(all_cases, args.pair_limit, args.pair_number)
    requested_caps = _requested_caps(args)
    if requested_caps != APPROVED_CAPS[len(cases)]:
        raise SystemExit(f"{len(cases)}쌍 후보 생성은 승인 cap과 정확히 같아야 합니다")

    settings = load_settings(PROVIDER_CONFIG)
    _require_approved_provider(settings)
    images = _validate_images(
        cases,
        max_bytes=settings.documents.model_image_max_bytes,
        max_width=settings.documents.model_image_max_width,
    )
    git_sha, git_clean = _git_state()
    if len(cases) == 30 and not git_clean:
        raise SystemExit("30쌍 후보 생성은 변경 사항이 없는 Git commit에서만 허용합니다")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"비어 있지 않은 출력 폴더입니다: {output}")

    prompt_hashes = {
        "baseline": _sha256(BASELINE_PROMPT),
        "improved": _sha256(IMPROVED_PROMPT),
    }
    input_hashes = {
        CASES: _sha256(CASES),
        SELECTION: _sha256(SELECTION),
        BASELINE_PROMPT: prompt_hashes["baseline"],
        IMPROVED_PROMPT: prompt_hashes["improved"],
        PROVIDER_CONFIG: _sha256(PROVIDER_CONFIG),
        PROJECT_ROOT / "uv.lock": _sha256(PROJECT_ROOT / "uv.lock"),
        **{images[case.pair_id]: case.image_sha256 for case in cases},
        **{
            PROJECT_ROOT / relative: _sha256(PROJECT_ROOT / relative)
            for relative in PROVENANCE_COMPONENTS
        },
    }
    call_context = {
        f"{case.pair_id}/{source}": {
            "candidate_source": source,
            "prompt_sha256": prompt_hashes[source],
            "input_sha256": task_input_sha256(case),
        }
        for case in cases
        for source in ("baseline", "improved")
    }

    load_project_env(PROJECT_ROOT)
    calls_path = output / "candidate-calls.jsonl"
    results_path = output / "candidate-results.jsonl"
    summary_path = output / "candidate-summary.json"
    last_journal_call: dict[str, Any] | None = None
    call_record_count = 0

    def record_call(record: dict[str, Any]) -> None:
        nonlocal call_record_count, last_journal_call
        enriched = {**record, **call_context.get(str(record.get("sample_id")), {})}
        _append_jsonl(calls_path, enriched)
        last_journal_call = enriched
        call_record_count += 1

    provider = LiteLLMProvider(
        model=settings.provider.model,
        expected_actual_model=settings.provider.expected_actual_model,
        api_key_env=settings.provider.api_key_env or "",
        api_base=settings.provider.api_base,
        structured_output=settings.provider.structured_output,
        max_requests=requested_caps.max_requests,
        max_attempts=requested_caps.max_attempts,
        requests_per_minute=settings.limits.requests_per_minute,
        max_retries=args.max_retries,
        retry_initial_seconds=settings.limits.retry_initial_seconds,
        max_cost_usd=requested_caps.max_cost_usd,
        max_input_tokens=requested_caps.max_input_tokens,
        max_output_tokens=requested_caps.max_output_tokens,
        max_wall_seconds=requested_caps.max_wall_seconds,
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
        max_images_per_prompt=1,
        on_response_received=record_call,
    )
    output.mkdir(parents=True, exist_ok=True)
    calls_path.write_text("", encoding="utf-8")
    baseline_prompt_bytes = BASELINE_PROMPT.read_bytes()
    improved_prompt_bytes = IMPROVED_PROMPT.read_bytes()
    (output / BASELINE_PROMPT.name).write_bytes(baseline_prompt_bytes)
    (output / IMPROVED_PROMPT.name).write_bytes(improved_prompt_bytes)

    drafts: list[CandidatePairDraft] = []
    run_error: Exception | None = None
    started_at = datetime.now(UTC)
    try:
        pairs = generate_candidate_pairs(
            cases=cases,
            project_root=PROJECT_ROOT,
            baseline_prompt_path=BASELINE_PROMPT,
            improved_prompt_path=IMPROVED_PROMPT,
            provider=provider,
            on_pair=drafts.append,
        )
    except Exception as exc:
        run_error = exc
        terminal_call = dict(provider.last_call or {})
        enriched_terminal = {
            **terminal_call,
            **call_context.get(str(terminal_call.get("sample_id")), {}),
        }
        if terminal_call and _attempt_identity(enriched_terminal) != _attempt_identity(
            last_journal_call or {}
        ):
            record_call(terminal_call)
        pairs = bind_candidate_set_sha256(drafts)

    expected_request_count = len(cases) * 2
    budget = provider.budget.summary()
    actual_request_count = int(budget.get("request_count", 0))
    actual_attempt_count = int(budget.get("attempt_count", 0))
    if run_error is None and (
        len(pairs) != len(cases)
        or actual_request_count != expected_request_count
        or actual_attempt_count != expected_request_count
        or call_record_count != expected_request_count
    ):
        run_error = RuntimeError(
            "후보 생성 기록이 불완전합니다: "
            f"pair {len(pairs)}/{len(cases)}, request {actual_request_count}/"
            f"{expected_request_count}, attempt {actual_attempt_count}/{expected_request_count}, "
            f"journal {call_record_count}/{expected_request_count}"
        )

    changed_inputs = []
    for path, expected_hash in input_hashes.items():
        try:
            if _sha256(path) != expected_hash:
                changed_inputs.append(path.name)
        except OSError:
            changed_inputs.append(path.name)
    if changed_inputs and run_error is None:
        run_error = RuntimeError("실행 중 입력이 변경됐습니다: " + ", ".join(changed_inputs))

    _write_results(results_path, pairs)
    probe_only = len(cases) == 1
    complete = run_error is None and len(pairs) == len(cases)
    observed_status = (
        "complete"
        if complete
        else "partial"
        if pairs or actual_request_count or call_record_count
        else "inconclusive"
    )
    actual_models = sorted(
        {
            provenance.actual_model
            for pair in pairs
            for provenance in (pair.candidate_a_provenance, pair.candidate_b_provenance)
        }
    )
    candidate_set_hash = pairs[0].candidate_set_sha256 if pairs else None
    invalid_output_count = sum(
        status == "invalid_output"
        for pair in pairs
        for status in (
            pair.candidate_a_validation_status,
            pair.candidate_b_validation_status,
        )
    )
    source_counts = Counter(case.course_split for case in cases)
    summary: dict[str, Any] = {
        "artifact_schema_version": 2,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "status": (
            "inconclusive"
            if probe_only or run_error
            else "fail"
            if invalid_output_count
            else "pass"
        ),
        "observed_status": observed_status,
        "probe_only": probe_only,
        "evidence_kind": "live_quality",
        "pair_count": len(cases),
        "pair_numbers": [all_cases.index(case) + 1 for case in cases],
        "pair_ids": [case.pair_id for case in cases],
        "completed_pair_count": len(pairs),
        "source_split_counts": dict(source_counts),
        "expected_request_count": expected_request_count,
        "actual_request_count": actual_request_count,
        "maximum_request_count": requested_caps.max_requests,
        "actual_attempt_count": actual_attempt_count,
        "maximum_attempt_count": requested_caps.max_attempts,
        "max_retries_per_request": args.max_retries,
        "git_sha": git_sha,
        "git_clean": git_clean,
        "requested_model": settings.provider.model,
        "expected_actual_model": settings.provider.expected_actual_model,
        "actual_models": actual_models,
        "provider": "nvidia_nim",
        "api_base": settings.provider.api_base,
        "api_key_env": settings.provider.api_key_env,
        "reference_sent_to_task_model": False,
        "catalog_verified_on": args.catalog_verified_on.isoformat(),
        "billing_basis": settings.provider.billing_basis,
        "pricing_source_url": settings.provider.pricing_source_url,
        "pricing_verified_on": args.pricing_verified_on.isoformat(),
        "cases_sha256": input_hashes[CASES],
        "selection_sha256": input_hashes[SELECTION],
        "task_input_sha256": {
            case.pair_id: task_input_sha256(case) for case in cases
        },
        "baseline_prompt_sha256": prompt_hashes["baseline"],
        "improved_prompt_sha256": prompt_hashes["improved"],
        "baseline_prompt_snapshot_sha256": _sha256(output / BASELINE_PROMPT.name),
        "improved_prompt_snapshot_sha256": _sha256(output / IMPROVED_PROMPT.name),
        "provider_config_sha256": input_hashes[PROVIDER_CONFIG],
        "lockfile_sha256": input_hashes[PROJECT_ROOT / "uv.lock"],
        "workflow_component_sha256": {
            relative: input_hashes[PROJECT_ROOT / relative]
            for relative in PROVENANCE_COMPONENTS
        },
        "candidate_set_sha256": candidate_set_hash,
        "candidate_results_sha256": _sha256(results_path),
        "candidate_calls_sha256": _sha256(calls_path),
        "candidate_call_record_count": call_record_count,
        "invalid_output_count": invalid_output_count,
        "budget": budget,
    }
    if changed_inputs:
        summary["input_changes"] = changed_inputs
    if run_error is not None:
        summary["error_type"] = type(run_error).__name__
        summary["error_message"] = str(run_error)
    atomic_write_json(summary_path, summary)

    if run_error is not None:
        print(
            f"OpenCQA 후보 생성이 중단됐습니다: "
            f"pair={len(pairs)}/{len(cases)}, "
            f"request={actual_request_count}/{expected_request_count}"
        )
        return 2
    print(
        f"OpenCQA {len(cases)}쌍 baseline·improved 후보를 "
        f"task model 요청 {actual_request_count}회로 생성했습니다"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
