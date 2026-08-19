# 목적: 개발 문제의 실패 답으로 새 지시문을 만들고 검증 문제에서 처음 지시문과 비교한다.
# 기대 결과: 후보·선택 지시문, 검증 점수, 두 모델의 호출 기록과 선택 이유가 저장된다.

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date, datetime
from pathlib import Path

from deepeval.prompt import Prompt

from verifiable_ai_workflow.config.secrets import load_project_env
from verifiable_ai_workflow.config.settings import load_settings
from verifiable_ai_workflow.course_live import build_course_provider, summarize_call_failures
from verifiable_ai_workflow.judge_model import CourseJudgeModel
from verifiable_ai_workflow.live_execution import LiveBudget, LiveBudgetCaps
from verifiable_ai_workflow.open_cqa_candidates import load_open_cqa_cases
from verifiable_ai_workflow.prompt_optimization import (
    OpenCqaDeterministicMetric,
    OpenCqaVlmCallback,
    build_prompt_optimizer,
    build_selection_source_evidence,
    score_output,
    split_goldens,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_CONFIG = PROJECT_ROOT / "configs/nvidia-nim-gemma4.yaml"
OPTIMIZER_CONFIG = PROJECT_ROOT / "configs/google-gemini-3.5-flash-lite-judge.yaml"
DEMO_CONFIG = PROJECT_ROOT / "configs/week-04-demo.yaml"
OPTIMIZER_REQUEST_OUTPUT_TOKEN_CEILING = 2_000
TARGET_APPROVED_CAPS = LiveBudgetCaps(
    max_requests=45,
    max_attempts=45,
    max_input_tokens=900_000,
    max_output_tokens=22_500,
    max_cost_usd=0.01,
    max_wall_seconds=7_200,
)
OPTIMIZER_APPROVED_CAPS = LiveBudgetCaps(
    max_requests=4,
    max_attempts=8,
    max_input_tokens=40_000,
    max_output_tokens=16_000,
    max_cost_usd=0.01,
    max_wall_seconds=7_200,
)
DEMO_TARGET_APPROVED_CAPS = LiveBudgetCaps(
    max_requests=5,
    max_attempts=5,
    max_input_tokens=100_000,
    max_output_tokens=2_500,
    max_cost_usd=0.01,
    max_wall_seconds=900,
)
DEMO_OPTIMIZER_APPROVED_CAPS = LiveBudgetCaps(
    max_requests=2,
    max_attempts=4,
    max_input_tokens=20_000,
    max_output_tokens=8_000,
    max_cost_usd=0.01,
    max_wall_seconds=900,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files_changed(expected: list[tuple[Path, str]]) -> bool:
    return any(not path.is_file() or _sha256(path) != digest for path, digest in expected)


def _output_path(output: Path | None, demo_samples: int | None) -> Path:
    if output is not None:
        return output
    if demo_samples:
        return PROJECT_ROOT / "reports/week-04" / f"class-demo-{datetime.now():%Y%m%d-%H%M%S}"
    raise ValueError("--output은 전체 평가에서 필요합니다")


def _artifact_sha256(output: Path, cases_path: Path) -> dict[str, str | None]:
    paths = {
        "calls.jsonl": output / "calls.jsonl",
        "validation.jsonl": output / "validation.jsonl",
        "candidate-prompt.md": output / "candidate-prompt.md",
        "selected-prompt.md": output / "selected-prompt.md",
        "week-03-cases.jsonl": cases_path,
    }
    return {name: _sha256(path) if path.is_file() else None for name, path in paths.items()}


def _clean_git() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise SystemExit("PromptOptimizer 실제 실행은 변경사항이 없는 Git commit에서만 허용합니다")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _select_prompt(
    baseline: Prompt,
    candidate: Prompt,
    baseline_mean: float,
    candidate_mean: float | None,
) -> tuple[str, Prompt, str]:
    if (candidate.text_template or "") == (baseline.text_template or ""):
        return "baseline", baseline, "candidate_identical"
    if candidate_mean is not None and candidate_mean > baseline_mean:
        return "candidate", candidate, "validation_improved"
    return "baseline", baseline, "validation_not_improved"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-optimize", action="store_true")
    parser.add_argument(
        "--demo-samples",
        type=int,
        choices=(2,),
        help="수업 중 GEPA 과정 시연에 사용할 development 사례 수(2건 고정)",
    )
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--catalog-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument("--pricing-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument("--optimizer-max-requests", type=int, required=True)
    parser.add_argument("--optimizer-max-attempts", type=int, required=True)
    parser.add_argument("--optimizer-max-input-tokens", type=int, required=True)
    parser.add_argument("--optimizer-max-output-tokens", type=int, required=True)
    parser.add_argument("--optimizer-max-cost-usd", type=float, required=True)
    parser.add_argument("--optimizer-max-wall-seconds", type=float, required=True)
    parser.add_argument("--optimizer-catalog-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument("--optimizer-pricing-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="결과 폴더. 2건 실습에서는 생략하면 실행 시각으로 자동 생성합니다",
    )
    args = parser.parse_args()
    try:
        args.output = _output_path(args.output, args.demo_samples)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.live_optimize:
        raise SystemExit("실제 최적화에는 --live-optimize가 필요합니다")
    for name, verified_on in (
        ("catalog", args.catalog_verified_on),
        ("pricing", args.pricing_verified_on),
        ("optimizer-catalog", args.optimizer_catalog_verified_on),
        ("optimizer-pricing", args.optimizer_pricing_verified_on),
    ):
        if verified_on != date.today():
            raise SystemExit(f"--{name}-verified-on은 실행 당일 날짜여야 합니다")
    target_caps = LiveBudgetCaps(
        max_requests=args.max_requests,
        max_attempts=args.max_requests,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_cost_usd=args.max_cost_usd,
        max_wall_seconds=args.max_wall_seconds,
    )
    optimizer_caps = LiveBudgetCaps(
        max_requests=args.optimizer_max_requests,
        max_attempts=args.optimizer_max_attempts,
        max_input_tokens=args.optimizer_max_input_tokens,
        max_output_tokens=args.optimizer_max_output_tokens,
        max_cost_usd=args.optimizer_max_cost_usd,
        max_wall_seconds=args.optimizer_max_wall_seconds,
    )
    approved_caps = (
        (DEMO_TARGET_APPROVED_CAPS, DEMO_OPTIMIZER_APPROVED_CAPS)
        if args.demo_samples
        else (TARGET_APPROVED_CAPS, OPTIMIZER_APPROVED_CAPS)
    )
    if (target_caps, optimizer_caps) != approved_caps:
        raise SystemExit("Week 4 PromptOptimizer는 두 provider의 승인 cap과 정확히 같아야 합니다")
    target_settings = load_settings(TARGET_CONFIG)
    optimizer_settings = load_settings(OPTIMIZER_CONFIG)
    git_sha = _clean_git()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"비어 있지 않은 출력 폴더입니다: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"결과 폴더: {args.output}")

    load_project_env(PROJECT_ROOT)
    calls_path = args.output / "calls.jsonl"
    call_records: list[dict] = []
    last_journal_calls: dict[str, dict] = {}

    def record_call(role: str, call: dict) -> None:
        record = {**call, "provider_role": role}
        call_records.append(record)
        with calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        last_journal_calls[role] = record

    target_provider = build_course_provider(
        target_settings,
        target_caps,
        structured_output="json_schema",
        on_response=lambda call: record_call("target", call),
        budget=LiveBudget(target_caps),
    )
    optimizer_provider = build_course_provider(
        optimizer_settings,
        optimizer_caps,
        structured_output="json_schema",
        request_output_token_ceiling=OPTIMIZER_REQUEST_OUTPUT_TOKEN_CEILING,
        on_response=lambda call: record_call("optimizer", call),
        budget=LiveBudget(optimizer_caps),
    )
    callback = OpenCqaVlmCallback(target_provider, PROJECT_ROOT)
    cases_path = PROJECT_ROOT / "local-data/opencqa/week-03-cases.jsonl"
    cases = load_open_cqa_cases(cases_path)
    cases_sha256 = _sha256(cases_path)
    splits = split_goldens(cases)
    source_evidence = build_selection_source_evidence(cases, splits)
    run_goldens = (
        splits["development"][: args.demo_samples] if args.demo_samples else splits["development"]
    )
    optimizer_config = DEMO_CONFIG if args.demo_samples else PROJECT_ROOT / "configs/week-04.yaml"
    baseline_path = PROJECT_ROOT / "prompts/week-04-baseline.md"
    baseline = Prompt(text_template=baseline_path.read_text(encoding="utf-8"))
    source_paths = {
        "baseline_prompt_sha256": baseline_path,
        "schema_sha256": PROJECT_ROOT / "src/verifiable_ai_workflow/schemas/models.py",
        "scorer_sha256": PROJECT_ROOT / "src/verifiable_ai_workflow/prompt_optimization.py",
        "optimizer_config_sha256": optimizer_config,
        "target_provider_config_sha256": TARGET_CONFIG,
        "optimizer_provider_config_sha256": OPTIMIZER_CONFIG,
    }
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    expected_files = [
        (cases_path, cases_sha256),
        *((path, source_hashes[name]) for name, path in source_paths.items()),
        *((PROJECT_ROOT / case.image_path, case.image_sha256) for case in cases),
    ]

    def artifact_hashes() -> dict[str, str | None]:
        hashes = _artifact_sha256(args.output, cases_path)
        hashes["week-03-cases.jsonl"] = cases_sha256
        return hashes

    optimizer = build_prompt_optimizer(
        goldens=run_goldens,
        model_callback=callback,
        optimizer_model=CourseJudgeModel(optimizer_provider),
        config_path=optimizer_config,
    )

    def provider_evidence(role, provider, settings, catalog_date, pricing_date) -> dict:
        calls = [call for call in call_records if call["provider_role"] == role]
        errors, drifts = summarize_call_failures(calls, provider.expected_actual_model)
        return {
            "role": role,
            "requested_model": provider.model,
            "expected_actual_model": provider.expected_actual_model,
            "actual_models": sorted(
                {
                    str(call["actual_model"])
                    for call in calls
                    if call.get("actual_model") is not None
                }
            ),
            "provider_error_count": errors,
            "model_drift_count": drifts,
            "structured_output": provider.structured_output,
            "request_output_token_ceiling": provider.request_output_token_ceiling,
            "billing_basis": settings.provider.billing_basis,
            "pricing_source_url": settings.provider.pricing_source_url,
            "input_cost_per_token_usd": settings.provider.input_cost_per_token_usd,
            "output_cost_per_token_usd": settings.provider.output_cost_per_token_usd,
            "catalog_verified_on": catalog_date.isoformat(),
            "pricing_verified_on": pricing_date.isoformat(),
            "budget": provider.budget.summary(),
        }

    def save_inconclusive(exc: Exception) -> int:
        for role, provider in (
            ("target", target_provider),
            ("optimizer", optimizer_provider),
        ):
            terminal_call = dict(provider.last_call or {})
            recorded = {**terminal_call, "provider_role": role}
            if terminal_call and recorded != last_journal_calls.get(role):
                record_call(role, terminal_call)
        target_evidence = provider_evidence(
            "target",
            target_provider,
            target_settings,
            args.catalog_verified_on,
            args.pricing_verified_on,
        )
        optimizer_evidence = provider_evidence(
            "optimizer",
            optimizer_provider,
            optimizer_settings,
            args.optimizer_catalog_verified_on,
            args.optimizer_pricing_verified_on,
        )
        attempt_count = target_evidence["budget"].get("attempt_count", 0) + optimizer_evidence[
            "budget"
        ].get("attempt_count", 0)
        input_changed = _files_changed(expected_files)
        summary = {
            "status": "inconclusive",
            "observed_status": "partial" if attempt_count else "not_run",
            "evidence_kind": "live_quality",
            "git_sha": git_sha,
            "run_mode": "classroom_demo" if args.demo_samples else "full_evaluation",
            "demo_sample_count": args.demo_samples,
            "quality_selection_allowed": not bool(args.demo_samples),
            "development_count": 18,
            "validation_count": 6,
            "test_count": 6,
            **source_evidence,
            "target_provider": target_evidence,
            "optimizer_provider": optimizer_evidence,
            "provider_error_count": (
                target_evidence["provider_error_count"] + optimizer_evidence["provider_error_count"]
            ),
            "model_drift_count": (
                target_evidence["model_drift_count"] + optimizer_evidence["model_drift_count"]
            ),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "input_changed_during_run": input_changed,
            **source_hashes,
            "artifact_sha256": artifact_hashes(),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"PromptOptimizer 실행이 중단됐습니다: {type(exc).__name__}")
        return 2

    try:
        candidate = optimizer.optimize(baseline, run_goldens)
    except Exception as exc:
        return save_inconclusive(exc)
    (args.output / "candidate-prompt.md").write_text(
        candidate.text_template or "",
        encoding="utf-8",
    )

    if args.demo_samples:
        target_evidence = provider_evidence(
            "target",
            target_provider,
            target_settings,
            args.catalog_verified_on,
            args.pricing_verified_on,
        )
        optimizer_evidence = provider_evidence(
            "optimizer",
            optimizer_provider,
            optimizer_settings,
            args.optimizer_catalog_verified_on,
            args.optimizer_pricing_verified_on,
        )
        provider_errors = (
            target_evidence["provider_error_count"] + optimizer_evidence["provider_error_count"]
        )
        model_drifts = (
            target_evidence["model_drift_count"] + optimizer_evidence["model_drift_count"]
        )
        input_changed = _files_changed(expected_files)
        attempt_count = target_evidence["budget"].get("attempt_count", 0) + optimizer_evidence[
            "budget"
        ].get("attempt_count", 0)
        summary = {
            "status": (
                "inconclusive" if provider_errors or model_drifts or input_changed else "pass"
            ),
            "observed_status": (
                "complete" if not input_changed else "partial" if attempt_count else "not_run"
            ),
            "evidence_kind": "live_quality",
            "git_sha": git_sha,
            "run_mode": "classroom_demo",
            "recommended_use": "prompt_optimization_process_only",
            "quality_selection_allowed": False,
            "demo_sample_count": args.demo_samples,
            "demo_sample_ids": [
                (golden.additional_metadata or {})["sample_id"] for golden in run_goldens
            ],
            "development_count": 18,
            "validation_count": 6,
            "test_count": 6,
            **source_evidence,
            "test_used_for_generation_or_selection": False,
            "candidate_changed": (candidate.text_template or "") != (baseline.text_template or ""),
            "selected": None,
            "selection_reason": "not_evaluated_demo",
            "target_provider": target_evidence,
            "optimizer_provider": optimizer_evidence,
            "provider_error_count": provider_errors,
            "model_drift_count": model_drifts,
            "input_changed_during_run": input_changed,
            **source_hashes,
            "artifact_sha256": artifact_hashes(),
        }
        if input_changed:
            summary["error_type"] = "InputChangedDuringRun"
            summary["error_message"] = "실행 중 지시문 또는 입력 파일이 바뀌었습니다"
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"지시문 최적화 실습 {args.demo_samples}건 완료: "
            f"후보 생성={summary['candidate_changed']}, "
            "품질 선택=하지 않음"
        )
        return 0 if summary["status"] == "pass" else 2

    metric = OpenCqaDeterministicMetric()
    records: list[dict] = []
    try:
        candidate_changed = (candidate.text_template or "") != (baseline.text_template or "")
        for golden in splits["validation"]:
            prompts = [("baseline", baseline)]
            if candidate_changed:
                prompts.append(("candidate", candidate))
            for name, prompt in prompts:
                output = callback(prompt, golden)
                records.append(
                    {
                        "prompt": name,
                        "output": output,
                        **score_output(metric, golden, output),
                    }
                )
    except Exception as exc:
        return save_inconclusive(exc)
    (args.output / "validation.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )

    def mean(name: str) -> float:
        values = [item["score"] for item in records if item["prompt"] == name]
        return sum(values) / len(values)

    baseline_mean = mean("baseline")
    candidate_mean = mean("candidate") if candidate_changed else None
    selected, selected_prompt, selection_reason = _select_prompt(
        baseline, candidate, baseline_mean, candidate_mean
    )
    selected_path = args.output / "selected-prompt.md"
    selected_path.write_text(selected_prompt.text_template or "", encoding="utf-8")
    target_evidence = provider_evidence(
        "target",
        target_provider,
        target_settings,
        args.catalog_verified_on,
        args.pricing_verified_on,
    )
    optimizer_evidence = provider_evidence(
        "optimizer",
        optimizer_provider,
        optimizer_settings,
        args.optimizer_catalog_verified_on,
        args.optimizer_pricing_verified_on,
    )
    provider_errors = (
        target_evidence["provider_error_count"] + optimizer_evidence["provider_error_count"]
    )
    model_drifts = target_evidence["model_drift_count"] + optimizer_evidence["model_drift_count"]
    input_changed = _files_changed(expected_files)
    attempt_count = target_evidence["budget"].get("attempt_count", 0) + optimizer_evidence[
        "budget"
    ].get("attempt_count", 0)
    summary = {
        "status": ("inconclusive" if provider_errors or model_drifts or input_changed else "pass"),
        "observed_status": (
            "complete" if not input_changed else "partial" if attempt_count else "not_run"
        ),
        "evidence_kind": "live_quality",
        "git_sha": git_sha,
        "run_mode": "full_evaluation",
        "development_count": 18,
        "validation_count": 6,
        "test_count": 6,
        **source_evidence,
        "test_used_for_generation_or_selection": False,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "candidate_changed": candidate_changed,
        "selected": selected,
        "selection_reason": selection_reason,
        "selected_prompt_sha256": _sha256(selected_path),
        "target_provider": target_evidence,
        "optimizer_provider": optimizer_evidence,
        "provider_error_count": provider_errors,
        "model_drift_count": model_drifts,
        "input_changed_during_run": input_changed,
        **source_hashes,
        "artifact_sha256": artifact_hashes(),
    }
    if input_changed:
        summary["error_type"] = "InputChangedDuringRun"
        summary["error_message"] = "실행 중 지시문 또는 입력 파일이 바뀌었습니다"
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"validation 평균 baseline={baseline_mean:.3f}, "
        f"candidate={candidate_mean if candidate_mean is not None else 'not_run'}, "
        f"선택={summary['selected']} ({selection_reason})"
    )
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
