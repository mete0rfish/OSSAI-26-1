"""Gemma 답 두 개를 Gemini Judge로 A/B·B/A 순서에서 두 번 비교한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import yaml
from PIL import Image

from verifiable_ai_workflow.config.secrets import load_project_env
from verifiable_ai_workflow.config.settings import load_settings
from verifiable_ai_workflow.judge_comparison import (
    load_complete_candidate_run,
    load_individual_human_label,
    validate_individual_human_label,
)
from verifiable_ai_workflow.judge_metrics import build_arena_metric, measure
from verifiable_ai_workflow.judge_model import CourseJudgeModel
from verifiable_ai_workflow.live_execution import LiveBudgetCaps
from verifiable_ai_workflow.open_cqa_candidates import candidate_set_sha256
from verifiable_ai_workflow.providers.litellm_provider import LiteLLMProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/week-03-judge.yaml"
IMAGES = (PROJECT_ROOT / "local-data/opencqa/images").resolve()
STUDENT_LABEL_DIRECTORY = Path("local-data/week-03-student-judges")
STUDENT_LABEL_ROOT = (PROJECT_ROOT / STUDENT_LABEL_DIRECTORY).resolve()
APPROVED_PROVIDER_CONFIG = "configs/google-gemini-3.5-flash-lite-judge.yaml"
APPROVED_RUBRIC = "configs/week-03-judge-rubric.yaml"
APPROVED_PROVIDER = {
    "model": "gemini/gemini-3.5-flash-lite",
    "expected_actual_model": "gemini-3.5-flash-lite",
    "api_base": "https://generativelanguage.googleapis.com/v1beta",
    "api_key_env": "GEMINI_API_KEY",
    "structured_output": "json_schema",
    "sampling_parameters": "omit",
    "billing_basis": "free_tier",
    "pricing_source_url": "https://ai.google.dev/gemini-api/docs/pricing",
    "input_cost_per_token_usd": 0.0,
    "output_cost_per_token_usd": 0.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return sha, dirty


def maximum_requests(pair_count: int) -> int:
    return pair_count * 2 * 2 * 2


def _select_pairs(all_pairs, pair_limit: int, pair_number: int):
    if len(all_pairs) != 30:
        raise SystemExit("candidate-results.jsonl에는 정확히 30쌍이 필요합니다")
    if pair_limit not in {1, 30}:
        raise SystemExit("--pair-limit는 승인된 1 또는 30이어야 합니다")
    if pair_limit == 30:
        if pair_number != 1:
            raise SystemExit("30쌍 실행에서는 --pair-number 1만 사용할 수 있습니다")
        return all_pairs
    if not 1 <= pair_number <= len(all_pairs):
        raise SystemExit(f"--pair-number는 1부터 {len(all_pairs)}까지입니다")
    return [all_pairs[pair_number - 1]]


def _changed_inputs(expected: dict[Path, str]) -> list[str]:
    return [path.name for path, digest in expected.items() if _sha256(path) != digest]


def _load_locked_human_label(path: Path, all_pairs, candidate_set_hash: str):
    resolved = path.resolve()
    if not resolved.is_relative_to(STUDENT_LABEL_ROOT) or not resolved.is_file():
        raise SystemExit("사람 사전 label은 local-data/week-03-student-judges 아래에 둡니다")
    try:
        human_label = load_individual_human_label(resolved)
        pair = validate_individual_human_label(
            human_label,
            all_pairs,
            candidate_set_hash,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(str(exc)) from exc
    return resolved, human_label, pair


def _validate_pair_images(
    pairs,
    *,
    max_bytes: int,
    max_width: int,
) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for pair in pairs:
        path = (PROJECT_ROOT / pair.image_path).resolve()
        if not path.is_relative_to(IMAGES) or not path.is_file():
            raise SystemExit(f"OpenCQA image_path가 준비 폴더의 파일이 아닙니다: {pair.pair_id}")
        if _sha256(path) != pair.image_sha256:
            raise SystemExit(f"OpenCQA 이미지 hash가 준비 기록과 다릅니다: {pair.pair_id}")
        if path.suffix.casefold() not in {".jpg", ".jpeg"} or path.stat().st_size > max_bytes:
            raise SystemExit(
                f"OpenCQA 이미지는 {max_bytes} bytes 이하 JPEG여야 합니다: {pair.pair_id}"
            )
        try:
            with Image.open(path) as image:
                if image.format != "JPEG" or image.width > max_width:
                    raise SystemExit(
                        f"OpenCQA 이미지는 너비 {max_width}px 이하 JPEG여야 합니다: "
                        f"{pair.pair_id}"
                    )
                image.verify()
        except OSError as exc:
            raise SystemExit(f"OpenCQA JPEG 이미지를 읽을 수 없습니다: {pair.pair_id}") from exc
        images[pair.pair_id] = path
    return images


def _load_approved_settings():
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if settings.get("provider_config") != APPROVED_PROVIDER_CONFIG:
        raise SystemExit("Week 3 Judge는 승인된 provider config만 사용합니다")
    if settings.get("rubric") != APPROVED_RUBRIC:
        raise SystemExit("Week 3 Judge는 승인된 rubric만 사용합니다")
    if settings.get("structured_output") != "json_schema":
        raise SystemExit("Week 3 Judge는 승인된 json_schema 출력만 사용합니다")
    provider_config = PROJECT_ROOT / APPROVED_PROVIDER_CONFIG
    provider_settings = load_settings(provider_config)
    actual = {
        field: getattr(provider_settings.provider, field) for field in APPROVED_PROVIDER
    }
    if actual != APPROVED_PROVIDER:
        raise SystemExit("Week 3 Judge는 승인된 Google endpoint·key·model·단가만 사용합니다")
    expected_limits = {
        "max_requests": 240,
        "requests_per_minute": 15,
        "max_retries": 1,
        "max_cost_usd": 0.01,
        "max_input_tokens": 1_200_000,
        "max_output_tokens": 120_000,
        "max_wall_seconds": 10_800,
        "request_input_token_ceiling": 5_000,
        "request_output_token_ceiling": 500,
    }
    actual_limits = {
        field: getattr(provider_settings.limits, field) for field in expected_limits
    }
    if actual_limits != expected_limits:
        raise SystemExit("Week 3 Judge provider config의 전체·요청별 cap이 승인값과 다릅니다")
    return (
        provider_settings,
        provider_config,
        PROJECT_ROOT / APPROVED_RUBRIC,
        settings["structured_output"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-judge", action="store_true")
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--pair-limit", type=int, required=True)
    parser.add_argument("--pair-number", type=int, default=1)
    parser.add_argument("--human-label", type=Path)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-retries", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--catalog-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument("--pricing-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.live_judge:
        raise SystemExit("실제 Judge 호출에는 --live-judge가 필요합니다")
    for name, verified_on in (
        ("catalog", args.catalog_verified_on),
        ("pricing", args.pricing_verified_on),
    ):
        if verified_on != date.today():
            raise SystemExit(f"--{name}-verified-on은 실행 당일 날짜여야 합니다")
    if args.max_retries != 1:
        raise SystemExit("Week 3 Judge는 일시적 429·5xx 오류를 요청당 1회만 재시도합니다")

    try:
        all_pairs, candidate_summary, candidate_paths, candidate_hashes = (
            load_complete_candidate_run(args.candidate_run, PROJECT_ROOT)
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"candidate run을 확인하세요: {exc}") from exc
    candidate_results_sha256 = candidate_hashes["candidate_results"]
    candidate_set_hash = candidate_set_sha256(all_pairs)
    pairs = _select_pairs(all_pairs, args.pair_limit, args.pair_number)
    approved_caps = {
        1: LiveBudgetCaps(
            max_requests=8,
            max_attempts=8,
            max_input_tokens=40_000,
            max_output_tokens=4_000,
            max_cost_usd=0.01,
            max_wall_seconds=300,
        ),
        30: LiveBudgetCaps(
            max_requests=240,
            max_attempts=240,
            max_input_tokens=1_200_000,
            max_output_tokens=120_000,
            max_cost_usd=0.01,
            max_wall_seconds=10_800,
        ),
    }
    requested = LiveBudgetCaps(
        max_requests=args.max_requests,
        max_attempts=args.max_requests,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_cost_usd=args.max_cost_usd,
        max_wall_seconds=args.max_wall_seconds,
    )
    if requested != approved_caps[len(pairs)]:
        raise SystemExit(f"{len(pairs)}쌍 Judge 실행은 문서의 승인 cap과 정확히 같아야 합니다")

    human_label_path = None
    human_label = None
    human_label_pair = None
    human_label_sha256 = None
    if args.human_label is not None:
        human_label_path, human_label, human_label_pair = _load_locked_human_label(
            args.human_label,
            all_pairs,
            candidate_set_hash,
        )
        human_label_sha256 = _sha256(human_label_path)
    if len(pairs) == 1:
        if human_label is None:
            raise SystemExit("1쌍 Judge 실행에는 먼저 작성한 --human-label이 필요합니다")
        if human_label_pair.pair_id != pairs[0].pair_id:
            raise SystemExit("사람 사전 label의 pair_number·pair_id가 선택한 pair와 다릅니다")
    elif human_label is not None:
        raise SystemExit("30쌍 Judge 실행의 사람 label은 compare 단계에서 연결합니다")

    provider_settings, provider_config, rubric, structured_output = (
        _load_approved_settings()
    )
    config_sha256 = _sha256(CONFIG)
    provider_config_sha256 = _sha256(provider_config)
    rubric_sha256 = _sha256(rubric)
    pair_images = _validate_pair_images(
        pairs,
        max_bytes=provider_settings.documents.model_image_max_bytes,
        max_width=provider_settings.documents.model_image_max_width,
    )
    git_sha, git_dirty = _git_state()
    if candidate_summary.get("git_sha") != git_sha:
        raise SystemExit("candidate run과 Judge 실행의 Git SHA가 다릅니다")
    if len(pairs) == 30 and git_dirty:
        raise SystemExit("30쌍 품질 실행은 변경사항이 없는 Git commit에서만 허용합니다")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise SystemExit(f"비어 있지 않은 출력 폴더입니다: {args.output}")

    load_project_env(PROJECT_ROOT)
    calls_path = args.output / "judge-calls.jsonl"
    last_journal_call: dict | None = None
    call_records: list[dict] = []

    def record_call(call: dict) -> None:
        nonlocal last_journal_call
        with calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(call, ensure_ascii=False) + "\n")
        last_journal_call = call
        call_records.append(call)

    provider = LiteLLMProvider(
        model=provider_settings.provider.model,
        expected_actual_model=provider_settings.provider.expected_actual_model,
        api_key_env=provider_settings.provider.api_key_env or "",
        api_base=provider_settings.provider.api_base,
        structured_output=structured_output,
        max_requests=args.max_requests,
        requests_per_minute=provider_settings.limits.requests_per_minute,
        max_retries=args.max_retries,
        max_attempts=args.max_requests,
        retry_initial_seconds=provider_settings.limits.retry_initial_seconds,
        max_cost_usd=args.max_cost_usd,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_wall_seconds=args.max_wall_seconds,
        request_input_token_ceiling=provider_settings.limits.request_input_token_ceiling,
        request_output_token_ceiling=provider_settings.limits.request_output_token_ceiling,
        request_timeout_seconds=provider_settings.limits.request_timeout_seconds,
        input_cost_per_token_usd=provider_settings.provider.input_cost_per_token_usd,
        output_cost_per_token_usd=provider_settings.provider.output_cost_per_token_usd,
        temperature=provider_settings.provider.temperature,
        top_p=provider_settings.provider.top_p,
        seed=provider_settings.provider.seed,
        sampling_parameters=provider_settings.provider.sampling_parameters,
        thinking_mode=provider_settings.provider.thinking_mode,
        thinking_parameter=provider_settings.provider.thinking_parameter,
        max_images_per_prompt=provider_settings.provider.max_images_per_prompt,
        on_response_received=record_call,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    model = CourseJudgeModel(provider, max_validation_retries=args.max_retries)
    metric = build_arena_metric(model, rubric)
    results_path = args.output / "judge-results.jsonl"
    result_count = 0
    expected_request_count = 0
    run_error: Exception | None = None
    try:
        for pair in pairs:
            model.image_path = pair_images[pair.pair_id]
            random_seed = int(hashlib.sha256(pair.pair_id.encode()).hexdigest()[:8], 16)
            for trial in (1, 2):
                model.call_id = f"{pair.pair_id}/trial-{trial}/ab"
                winner_ab, reason_ab = measure(
                    metric,
                    pair,
                    random_seed=random_seed,
                    max_retries=args.max_retries,
                )
                expected_request_count += 1 if winner_ab == "tie" else 2
                model.call_id = f"{pair.pair_id}/trial-{trial}/ba"
                winner_ba, reason_ba = measure(
                    metric,
                    pair,
                    reverse=True,
                    random_seed=random_seed,
                    max_retries=args.max_retries,
                )
                expected_request_count += 1 if winner_ba == "tie" else 2
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "pair_id": pair.pair_id,
                                "trial": trial,
                                "winner_ab": winner_ab,
                                "reason_ab": reason_ab,
                                "winner_ba": winner_ba,
                                "reason_ba": reason_ba,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                result_count += 1
    except Exception as exc:
        run_error = exc
        terminal_call = dict(provider.last_call or {})
        if terminal_call and terminal_call != last_journal_call:
            record_call(terminal_call)

    expected_trial_count = len(pairs) * 2
    expected_request_count += (
        model.structured_output_retry_count + model.invalid_winner_retry_count
    )
    budget = provider.budget.summary()
    actual_request_count = int(budget.get("request_count", 0))
    if run_error is None and (
        result_count != expected_trial_count
        or actual_request_count != expected_request_count
        or len(call_records) != actual_request_count
    ):
        run_error = RuntimeError(
            "Judge 호출 기록이 불완전합니다: "
            f"trial {result_count}/{expected_trial_count}, "
            f"request {actual_request_count}/{expected_request_count}, "
            f"journal {len(call_records)}/{actual_request_count}"
        )

    input_hashes = {
        **{candidate_paths[name]: digest for name, digest in candidate_hashes.items()},
        CONFIG: config_sha256,
        provider_config: provider_config_sha256,
        rubric: rubric_sha256,
        **{pair_images[pair.pair_id]: pair.image_sha256 for pair in pairs},
        **({human_label_path: human_label_sha256} if human_label_path else {}),
    }
    try:
        changed_inputs = _changed_inputs(input_hashes)
    except OSError as exc:
        changed_inputs = [str(exc)]
    if changed_inputs and run_error is None:
        run_error = RuntimeError(
            "Judge 실행 중 입력 파일이 변경되었습니다: " + ", ".join(changed_inputs)
        )

    probe_only = len(pairs) < 30
    observed_status = (
        "complete"
        if run_error is None
        else "partial"
        if result_count or actual_request_count or call_records
        else "inconclusive"
    )
    summary = {
        "status": "inconclusive" if probe_only or run_error else "pass",
        "observed_status": observed_status,
        "probe_only": probe_only,
        "evidence_kind": "live_quality",
        "pair_count": len(pairs),
        "pair_numbers": [all_pairs.index(pair) + 1 for pair in pairs],
        "pair_ids": [pair.pair_id for pair in pairs],
        "completed_pair_count": result_count // 2,
        "completed_trial_count": result_count,
        "expected_request_count": expected_request_count,
        "maximum_request_count": maximum_requests(len(pairs)),
        "actual_request_count": actual_request_count,
        "max_retries_per_request": args.max_retries,
        "maximum_attempt_count": args.max_requests,
        "actual_attempt_count": int(budget.get("attempt_count", 0)),
        "structured_output_retry_count": model.structured_output_retry_count,
        "invalid_winner_retry_count": model.invalid_winner_retry_count,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "model": provider.model,
        "expected_actual_model": provider.expected_actual_model,
        "catalog_verified_on": args.catalog_verified_on.isoformat(),
        "billing_basis": provider_settings.provider.billing_basis,
        "pricing_source_url": provider_settings.provider.pricing_source_url,
        "pricing_verified_on": args.pricing_verified_on.isoformat(),
        "input_cost_per_token_usd": provider_settings.provider.input_cost_per_token_usd,
        "output_cost_per_token_usd": provider_settings.provider.output_cost_per_token_usd,
        "sampling_parameters": provider_settings.provider.sampling_parameters,
        "candidate_run_directory": str(args.candidate_run.resolve()),
        "candidate_git_sha": candidate_summary["git_sha"],
        "candidate_run_validated_complete": True,
        "candidate_status": candidate_summary["status"],
        "candidate_invalid_output_count": candidate_summary["invalid_output_count"],
        "candidate_summary_sha256": candidate_hashes["candidate_summary"],
        "candidate_calls_sha256": candidate_hashes["candidate_calls"],
        "candidate_baseline_prompt_snapshot_sha256": candidate_hashes[
            "baseline_prompt_snapshot"
        ],
        "candidate_improved_prompt_snapshot_sha256": candidate_hashes[
            "improved_prompt_snapshot"
        ],
        "candidate_results_sha256": candidate_results_sha256,
        "candidate_set_sha256": candidate_set_hash,
        "reference_answer_role": "arena_expected_output",
        "human_calibrated": False,
        "judge_results_sha256": _sha256(results_path) if results_path.is_file() else None,
        "judge_calls_sha256": _sha256(calls_path) if calls_path.is_file() else None,
        "config_sha256": config_sha256,
        "provider_config_sha256": provider_config_sha256,
        "rubric_sha256": rubric_sha256,
        "rubric_path": rubric.relative_to(PROJECT_ROOT).as_posix(),
        "rubric_kind": "course_approved",
        "human_label_path": (
            (STUDENT_LABEL_DIRECTORY / human_label_path.relative_to(STUDENT_LABEL_ROOT)).as_posix()
            if human_label_path
            else None
        ),
        "human_label_sha256": human_label_sha256,
        "human_label_pair_number": human_label.pair_number if human_label else None,
        "human_label_pair_id": human_label.pair_id if human_label else None,
        "individual_human_label": human_label.label if human_label else None,
        "human_reviewer_id": human_label.reviewer_id if human_label else None,
        "human_label_locked_before_judge": human_label is not None,
        "budget": budget,
    }
    if run_error is not None:
        summary["error_type"] = type(run_error).__name__
        summary["error_message"] = str(run_error)
    if changed_inputs:
        summary["input_changes"] = changed_inputs
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if run_error:
        print(
            f"OpenCQA 실행이 중단됐습니다: 완료 pair={result_count // 2}, "
            f"완료 trial={result_count}"
        )
        return 2
    print(
        f"OpenCQA {len(pairs)}쌍 × 2회 × A/B·B/A를 "
        f"Judge 요청 {actual_request_count}/{maximum_requests(len(pairs))}회로 완료했습니다"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
