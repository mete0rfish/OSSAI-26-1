"""Gemma baseline/improved 후보에 대한 반복 Judge 결과를 비교한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from verifiable_ai_workflow.judge_comparison import (
    compare,
    load_complete_candidate_run,
    load_individual_human_label,
    load_judge_trials,
    validate_individual_human_label,
)
from verifiable_ai_workflow.open_cqa_candidates import candidate_set_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/week-03-judge.yaml"
APPROVED_PROVIDER_CONFIG = "configs/google-gemini-3.5-flash-lite-judge.yaml"
APPROVED_RUBRIC = "configs/week-03-judge-rubric.yaml"
APPROVED_MODEL = "gemini/gemini-3.5-flash-lite"
EXPECTED_ACTUAL_MODEL = "gemini-3.5-flash-lite"
STUDENT_LABEL_ROOT = (PROJECT_ROOT / "local-data/week-03-student-judges").resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_run_summary(run_summary: dict, source_sha256: dict[str, str]) -> None:
    if not run_summary:
        return
    bindings = {
        "candidate_results_sha256": "candidate_results",
        "candidate_set_sha256": "candidate_set",
        "candidate_summary_sha256": "candidate_summary",
        "candidate_calls_sha256": "candidate_calls",
        "candidate_baseline_prompt_snapshot_sha256": "baseline_prompt_snapshot",
        "candidate_improved_prompt_snapshot_sha256": "improved_prompt_snapshot",
        "judge_results_sha256": "judge_results",
        "judge_calls_sha256": "judge_calls",
        "config_sha256": "config",
        "provider_config_sha256": "provider_config",
        "rubric_sha256": "rubric",
    }
    if run_summary.get("human_label_path") is not None:
        bindings["human_label_sha256"] = "human_label"
    changed = [
        field
        for field, key in bindings.items()
        if run_summary.get(field) != source_sha256[key]
    ]
    if changed:
        raise SystemExit("summary.json과 현재 입력의 hash가 다릅니다: " + ", ".join(changed))


def _valid_judge_calls(
    calls: list[dict],
    actual_requests: int,
    actual_attempts: int,
    budget: dict,
    expected_sample_ids: set[str],
) -> bool:
    if any(not isinstance(call, dict) for call in calls):
        return False
    if len(calls) != actual_requests or [
        call.get("request_number") for call in calls
    ] != list(range(1, actual_requests + 1)):
        return False
    attempt_numbers = [call.get("attempt_number") for call in calls]
    if any(
        not isinstance(number, int) or not 1 <= number <= actual_attempts
        for number in attempt_numbers
    ) or len(set(attempt_numbers)) != len(attempt_numbers):
        return False
    if any(
        call.get("provider_status") != "provider_response_received"
        or call.get("actual_model") != EXPECTED_ACTUAL_MODEL
        or call.get("actual_model_matches_expected") is not True
        or not isinstance(call.get("raw_response"), dict)
        or not isinstance(call.get("sample_id"), str)
        or not call["sample_id"]
        or not isinstance(call.get("response_received_at"), str)
        or not call["response_received_at"]
        or not isinstance(call.get("input_tokens"), int)
        or call["input_tokens"] < 0
        or not isinstance(call.get("output_tokens"), int)
        or call["output_tokens"] < 0
        or not isinstance(call.get("actual_cost_usd"), (int, float))
        or call["actual_cost_usd"] < 0
        or not isinstance(call.get("latency_ms"), (int, float))
        or call["latency_ms"] < 0
        for call in calls
    ):
        return False
    return bool(
        {call["sample_id"] for call in calls} == expected_sample_ids
        and sum(call["input_tokens"] for call in calls)
        == budget.get("actual_input_tokens")
        and sum(call["output_tokens"] for call in calls)
        == budget.get("actual_output_tokens")
        and sum(call["actual_cost_usd"] for call in calls)
        == budget.get("actual_cost_usd")
        and sum(call["latency_ms"] for call in calls)
        <= budget.get("wall_seconds", -1) * 1000 + 1e-6
    )


def _is_live_quality(
    run_summary: dict,
    pair_count: int,
    source_sha256: dict[str, str],
    judge_calls: list[dict],
    expected_sample_ids: set[str],
) -> bool:
    actual_requests = run_summary.get("actual_request_count")
    actual_attempts = run_summary.get("actual_attempt_count")
    budget = run_summary.get("budget")
    return bool(
        pair_count == 30
        and run_summary.get("status") == "pass"
        and run_summary.get("observed_status") == "complete"
        and run_summary.get("probe_only") is False
        and run_summary.get("evidence_kind") == "live_quality"
        and run_summary.get("pair_count") == 30
        and run_summary.get("completed_pair_count") == 30
        and run_summary.get("completed_trial_count") == 60
        and isinstance(actual_requests, int)
        and 120 <= actual_requests <= 240
        and run_summary.get("expected_request_count") == actual_requests
        and run_summary.get("maximum_request_count") == 240
        and isinstance(actual_attempts, int)
        and actual_requests <= actual_attempts <= 240
        and run_summary.get("maximum_attempt_count") == 240
        and run_summary.get("max_retries_per_request") == 1
        and run_summary.get("billing_basis") == "free_tier"
        and run_summary.get("input_cost_per_token_usd") == 0.0
        and run_summary.get("output_cost_per_token_usd") == 0.0
        and run_summary.get("candidate_run_validated_complete") is True
        and isinstance(budget, dict)
        and budget.get("request_count") == actual_requests
        and budget.get("attempt_count") == actual_attempts
        and all(
            isinstance(budget.get(field), (int, float))
            and 0 <= budget[field] <= limit
            for field, limit in {
                "reserved_input_tokens": 1_200_000,
                "actual_input_tokens": 1_200_000,
                "charged_input_tokens": 1_200_000,
                "reserved_output_tokens": 120_000,
                "actual_output_tokens": 120_000,
                "charged_output_tokens": 120_000,
                "reserved_cost_usd": 0.01,
                "actual_cost_usd": 0.01,
                "charged_cost_usd": 0.01,
                "wall_seconds": 10_800,
            }.items()
        )
        and _valid_judge_calls(
            judge_calls,
            actual_requests,
            actual_attempts,
            budget,
            expected_sample_ids,
        )
        and run_summary.get("git_dirty") is False
        and run_summary.get("model") == APPROVED_MODEL
        and run_summary.get("expected_actual_model") == EXPECTED_ACTUAL_MODEL
        and run_summary.get("sampling_parameters") == "omit"
        and run_summary.get("reference_answer_role") == "arena_expected_output"
        and run_summary.get("candidate_results_sha256")
        == source_sha256["candidate_results"]
        and run_summary.get("candidate_set_sha256") == source_sha256["candidate_set"]
        and run_summary.get("judge_results_sha256") == source_sha256["judge_results"]
        and bool(source_sha256["judge_calls"])
        and run_summary.get("judge_calls_sha256") == source_sha256["judge_calls"]
        and run_summary.get("config_sha256") == source_sha256["config"]
        and run_summary.get("provider_config_sha256") == source_sha256["provider_config"]
        and run_summary.get("rubric_path", APPROVED_RUBRIC) == APPROVED_RUBRIC
        and run_summary.get("rubric_sha256") == source_sha256["rubric"]
    )


def _select_pairs(all_pairs, pair_limit: int | None, pair_number: int):
    if len(all_pairs) != 30:
        raise SystemExit("candidate-results.jsonl에는 정확히 30쌍이 필요합니다")
    if pair_limit is None:
        if pair_number != 1:
            raise SystemExit("--pair-number를 쓰려면 --pair-limit도 필요합니다")
        return all_pairs
    if pair_limit not in {1, 30}:
        raise SystemExit("--pair-limit는 1 또는 30이어야 합니다")
    if pair_limit == 30:
        if pair_number != 1:
            raise SystemExit("30쌍 비교에서는 --pair-number 1만 사용할 수 있습니다")
        return all_pairs
    if not 1 <= pair_number <= len(all_pairs):
        raise SystemExit(f"--pair-number는 1부터 {len(all_pairs)}까지입니다")
    return [all_pairs[pair_number - 1]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--judge-results", type=Path, required=True)
    parser.add_argument("--human-label", type=Path)
    parser.add_argument(
        "--human-label-sha256",
        help="Judge 결과 공개 전에 잠근 human-label.yaml SHA-256",
    )
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--pair-number", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/week-03/comparison.json",
    )
    args = parser.parse_args()
    try:
        all_pairs, candidate_summary, candidate_paths, candidate_hashes = (
            load_complete_candidate_run(args.candidate_run, PROJECT_ROOT)
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"candidate run을 확인하세요: {exc}") from exc
    candidate_set_hash = candidate_set_sha256(all_pairs)
    pairs = _select_pairs(all_pairs, args.pair_limit, args.pair_number)
    trials = load_judge_trials(args.judge_results)

    run_summary_path = args.judge_results.with_name("summary.json")
    run_summary = (
        json.loads(run_summary_path.read_text(encoding="utf-8"))
        if run_summary_path.is_file()
        else {}
    )
    if run_summary and (
        run_summary.get("pair_count") != len(pairs)
        or run_summary.get("pair_ids") != [pair.pair_id for pair in pairs]
        or run_summary.get("candidate_run_directory")
        != str(args.candidate_run.resolve())
        or run_summary.get("candidate_run_validated_complete") is not True
        or run_summary.get("candidate_git_sha") != candidate_summary.get("git_sha")
        or run_summary.get("candidate_status") != candidate_summary.get("status")
        or run_summary.get("candidate_invalid_output_count")
        != candidate_summary.get("invalid_output_count")
        or run_summary.get("git_sha") != candidate_summary.get("git_sha")
    ):
        raise SystemExit("summary.json의 candidate run·Git SHA·pair 범위가 비교 대상과 다릅니다")
    judge_calls_path = args.judge_results.with_name("judge-calls.jsonl")
    if run_summary.get("rubric_path", APPROVED_RUBRIC) != APPROVED_RUBRIC:
        raise SystemExit("summary.json은 과정의 고정 Judge rubric을 가리켜야 합니다")
    provider_config = PROJECT_ROOT / APPROVED_PROVIDER_CONFIG
    rubric = PROJECT_ROOT / APPROVED_RUBRIC
    for source in (CONFIG, provider_config, rubric):
        if not source.is_file():
            raise SystemExit(f"과정의 고정 Judge 설정이 없습니다: {source}")
    source_sha256 = {
        "candidate_results": candidate_hashes["candidate_results"],
        "candidate_set": candidate_set_hash,
        "candidate_summary": candidate_hashes["candidate_summary"],
        "candidate_calls": candidate_hashes["candidate_calls"],
        "baseline_prompt_snapshot": candidate_hashes["baseline_prompt_snapshot"],
        "improved_prompt_snapshot": candidate_hashes["improved_prompt_snapshot"],
        "judge_results": _sha256(args.judge_results),
        "judge_calls": _sha256(judge_calls_path) if judge_calls_path.is_file() else "",
        "config": _sha256(CONFIG),
        "provider_config": _sha256(provider_config),
        "rubric": _sha256(rubric),
    }

    human_label = None
    summary_human_label_value = run_summary.get("human_label_path")
    human_label_value = args.human_label or summary_human_label_value
    if human_label_value is not None:
        if not isinstance(human_label_value, (str, Path)):
            raise SystemExit("summary.json의 human_label_path를 확인하세요")
        human_label_path = Path(human_label_value)
        if not human_label_path.is_absolute():
            human_label_path = PROJECT_ROOT / human_label_path
        human_label_path = human_label_path.resolve()
        if (
            not human_label_path.is_relative_to(STUDENT_LABEL_ROOT)
            or not human_label_path.is_file()
        ):
            raise SystemExit("summary.json의 사람 사전 label 경로를 확인하세요")
        try:
            human_label = load_individual_human_label(human_label_path)
            validate_individual_human_label(
                human_label,
                all_pairs,
                candidate_set_hash,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if human_label.pair_id not in {pair.pair_id for pair in pairs}:
            raise SystemExit("사람 사전 label의 pair가 비교 대상에 없습니다")
        if summary_human_label_value is not None and (
            run_summary.get("human_label_pair_number") != human_label.pair_number
            or run_summary.get("human_label_pair_id") != human_label.pair_id
            or run_summary.get("individual_human_label") != human_label.label
            or run_summary.get("human_reviewer_id") != human_label.reviewer_id
            or run_summary.get("human_label_locked_before_judge") is not True
        ):
            raise SystemExit("사람 사전 label과 비교 대상 또는 summary.json이 다릅니다")
        source_sha256["human_label"] = _sha256(human_label_path)
    elif run_summary and len(pairs) == 1:
        raise SystemExit("1쌍 실행 summary.json에 사람 사전 label 기록이 없습니다")

    formal_full_comparison = len(pairs) == 30
    if formal_full_comparison and human_label is None:
        raise SystemExit("30쌍 사람 비교에는 --human-label이 필요합니다")
    if formal_full_comparison and args.human_label_sha256 is None:
        raise SystemExit("30쌍 사람 비교에는 --human-label-sha256이 필요합니다")
    if args.human_label_sha256 is not None and (
        human_label is None
        or args.human_label_sha256 != source_sha256.get("human_label")
    ):
        raise SystemExit("잠근 사람 사전 label SHA-256이 현재 파일과 다릅니다")

    _verify_run_summary(run_summary, source_sha256)
    try:
        judge_calls = (
            [
                json.loads(line)
                for line in judge_calls_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if judge_calls_path.is_file()
            else []
        )
    except (json.JSONDecodeError, OSError):
        judge_calls = []
    live_quality = _is_live_quality(
        run_summary,
        len(pairs),
        source_sha256,
        judge_calls,
        {
            f"{pair.pair_id}/trial-{trial}/{order}"
            for pair in pairs
            for trial in (1, 2)
            for order in ("ab", "ba")
        },
    )
    if formal_full_comparison and not live_quality:
        raise SystemExit("30쌍 사람 비교에는 완결된 live_quality Judge 결과가 필요합니다")
    summary = compare(
        pairs,
        trials,
        human_label=human_label,
        candidate_set_hash=candidate_set_hash,
        live_quality=live_quality,
    )
    summary = summary.model_copy(
        update={
            "candidate_invalid_output_count": candidate_summary["invalid_output_count"],
            "reasons": [
                *summary.reasons,
                *(
                    [f"task output contract violation={candidate_summary['invalid_output_count']}"]
                    if candidate_summary["invalid_output_count"]
                    else []
                ),
            ],
            "source_sha256": {
                **source_sha256,
                "judge_summary": (
                    _sha256(run_summary_path) if run_summary_path.is_file() else ""
                ),
            }
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"평가 쌍={summary.pair_count}, baseline 승={summary.baseline_wins}, "
        f"improved 승={summary.improved_wins}, tie={summary.ties}, "
        f"review={summary.reviews}, 사용={summary.recommended_use}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
