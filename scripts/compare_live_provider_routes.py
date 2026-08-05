#!/usr/bin/env python3
"""Canonical 40건을 승인된 실제 두 provider로 제한 비교한다."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from verifiable_ai_workflow.config import load_project_env
from verifiable_ai_workflow.live_execution import (
    LiveExecutionError,
    require_canonical_project_file,
)
from verifiable_ai_workflow.live_provider_comparison import (
    Week2LiveError,
    WholeRunCaps,
    load_week2_live_config,
    run_week2_live,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = "configs/week-02-live.yaml"
PROBE_SAMPLE_IDS = (
    "aihub-report-r01",  # 긴 문장에 질문의 연도를 반복한 사례
    "aihub-report-r03",  # JSON 두 개와 수정 설명을 반환한 사례
    "aihub-report-r31",  # 답변 보류 대신 표의 숫자를 추측한 사례
)


def _require_approved_routes(config) -> None:
    actual = {
        "baseline": {
            "provider_id": config.baseline_route.provider_id,
            "model": config.baseline_route.model,
            "expected_actual_model": config.baseline_route.expected_actual_model,
            "api_base": config.baseline_route.api_base,
            "api_key_env": config.baseline_route.api_key_env,
        },
        "candidate": {
            "provider_id": config.candidate_route.provider_id,
            "model": config.candidate_route.model,
            "expected_actual_model": config.candidate_route.expected_actual_model,
            "api_base": config.candidate_route.api_base,
            "api_key_env": config.candidate_route.api_key_env,
        },
    }
    expected = {
        "baseline": {
            "provider_id": "nvidia-nim",
            "model": "nvidia_nim/google/gemma-4-31b-it",
            "expected_actual_model": "google/gemma-4-31b-it",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NVIDIA_NIM_API_KEY",
        },
        "candidate": {
            "provider_id": "google-ai-studio",
            "model": "gemini/gemini-3.5-flash-lite",
            "expected_actual_model": "gemini-3.5-flash-lite",
            "api_base": "https://generativelanguage.googleapis.com/v1beta",
            "api_key_env": "GEMINI_API_KEY",
        },
    }
    if actual != expected:
        raise Week2LiveError(
            "Week 2 live 호출은 승인된 NVIDIA·Google AI Studio endpoint, "
            "key 환경 변수와 model만 허용합니다"
        )


def _require_approved_caps(args: argparse.Namespace) -> None:
    mode = "probe" if args.probe_sample_id is not None else "full"
    profiles = {
        "probe": {
            "max_requests": 2,
            "max_input_tokens": 40_000,
            "max_output_tokens": 1_000,
            "max_retries": 0,
            "max_cost_usd": 0.01,
            "max_wall_seconds": 240,
        },
        "full": {
            "max_requests": 80,
            "max_input_tokens": 1_600_000,
            "max_output_tokens": 40_000,
            "max_retries": 0,
            "max_cost_usd": 0.01,
            "max_wall_seconds": 3_600,
        },
    }
    mismatches = [
        f"{name}={getattr(args, name)} != approved={value}"
        for name, value in profiles[mode].items()
        if not math.isclose(getattr(args, name), value, rel_tol=0.0, abs_tol=1e-12)
    ]
    if mismatches:
        raise Week2LiveError(f"Week 2 {mode} 승인 cap과 다릅니다: " + "; ".join(mismatches))


def _probe_succeeded(execution) -> bool:
    return (
        execution.summary.baseline_observation_count == 1
        and execution.summary.candidate_observation_count == 1
        and execution.summary.baseline_provider_errors == 0
        and execution.summary.candidate_provider_errors == 0
        and execution.summary.baseline_invalid_outputs == 0
        and execution.summary.candidate_invalid_outputs == 0
        and execution.baseline_provenance.actual_model_mismatch_count == 0
        and execution.candidate_provenance.actual_model_mismatch_count == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 2 two-provider live comparison")
    parser.add_argument("--config", default=CANONICAL_CONFIG)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--max-retries", type=int, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--catalog-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--probe-sample-id",
        choices=PROBE_SAMPLE_IDS,
        help="두 provider에 canonical sample 1건씩만 보내고 inconclusive 증거를 남깁니다",
    )
    args = parser.parse_args()
    if not args.live:
        parser.error("실제 두-provider 호출에는 --live가 필요합니다")

    load_project_env(PROJECT_ROOT)
    try:
        config_path = require_canonical_project_file(
            PROJECT_ROOT,
            args.config,
            CANONICAL_CONFIG,
        )
        _require_approved_routes(load_week2_live_config(config_path))
        _require_approved_caps(args)
        caps = WholeRunCaps(
            max_requests=args.max_requests,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            max_cost_usd=args.max_cost_usd,
            max_wall_seconds=args.max_wall_seconds,
        )
        execution = run_week2_live(
            PROJECT_ROOT,
            config_path=config_path,
            caps=caps,
            catalog_verified_on=args.catalog_verified_on,
            output_dir=args.output,
            probe_sample_id=args.probe_sample_id,
        )
    except (
        ValidationError,
        Week2LiveError,
        LiveExecutionError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"Week 2 live 실행 차단: {exc}", file=sys.stderr)
        return 2

    print(execution.summary.model_dump_json(indent=2))
    if args.probe_sample_id is not None:
        return 0 if _probe_succeeded(execution) else 2
    if execution.summary.automated_status == "inconclusive":
        return 2
    return 1 if execution.summary.automated_status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
