"""Gemma 후보 쌍과 반복 Judge 결과를 비교한다."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .open_cqa_candidates import (
    OpenCQACandidatePair,
    candidate_set_sha256,
    load_candidate_pairs,
    validate_candidate_output,
)

Preference = Literal["candidate_a", "tie", "candidate_b"]
ResolvedWinner = Literal["baseline", "improved", "tie", "review"]
JudgePair = OpenCQACandidatePair
load_pairs = load_candidate_pairs


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JudgeTrial(StrictModel):
    pair_id: str
    trial: Literal[1, 2]
    winner_ab: Preference
    reason_ab: str
    winner_ba: Preference
    reason_ba: str


class IndividualHumanLabel(StrictModel):
    pair_number: int = Field(ge=1)
    pair_id: str = Field(min_length=1, max_length=100)
    candidate_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(min_length=1, max_length=100)
    label: Preference
    reason: str = Field(min_length=10, max_length=1_000)

    @model_validator(mode="after")
    def required_text_is_present(self) -> IndividualHumanLabel:
        if not self.pair_id.strip() or not self.reviewer_id.strip() or not self.reason.strip():
            raise ValueError("pair_id, reviewer_id와 reason이 필요합니다")
        return self


class PairAudit(StrictModel):
    pair_id: str
    candidate_a_source: Literal["baseline", "improved"]
    candidate_b_source: Literal["baseline", "improved"]
    judge_label: Preference | Literal["review"]
    winner: ResolvedWinner
    order_conflict: bool
    repetition_conflict: bool
    individual_human_label: Preference | None = None
    judge_agrees_with_individual_human: bool | None = None


class ComparisonSummary(StrictModel):
    pair_count: int = Field(ge=1)
    judge_execution_evidence_kind: Literal["exploratory", "live_quality"]
    candidate_invalid_output_count: int = Field(default=0, ge=0)
    baseline_wins: int = Field(ge=0)
    improved_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    reviews: int = Field(ge=0)
    order_conflicts: int = Field(ge=0)
    repetition_conflicts: int = Field(ge=0)
    individual_human_label_count: Literal[0, 1]
    individual_human_agreement: float | None = Field(default=None, ge=0, le=1)
    human_calibrated: Literal[False]
    blocking_eligible: Literal[False]
    recommended_use: Literal["classroom_demo"]
    reasons: list[str]
    pairs: list[PairAudit]
    source_sha256: dict[str, str] = Field(default_factory=dict)


def _jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"준비 파일이 없습니다: {source}")
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_response_content(call: dict) -> str:
    raw = call.get("raw_response")
    content = raw.get("content") if isinstance(raw, dict) else None
    if content is None and isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("candidate call에 원문 응답이 없습니다")
    return content


def _validate_candidate_execution(
    summary: dict,
    calls: list[dict],
    *,
    requested_model: str,
    actual_model: str,
) -> None:
    budget = summary.get("budget")
    if not isinstance(budget, dict):
        raise ValueError("candidate summary budget이 없습니다")
    expected = {
        "request_count": 60,
        "attempt_count": 60,
        "reserved_input_tokens": 1_200_000,
        "reserved_output_tokens": 30_000,
    }
    if (
        any(
            not isinstance(actual := budget.get(field), int)
            or isinstance(actual, bool)
            or actual != value
            for field, value in expected.items()
        )
        or budget.get("reserved_cost_usd") != 0.0
    ):
        raise ValueError("candidate budget의 요청·attempt·예약량이 승인값과 다릅니다")
    token_limits = {
        "actual_input_tokens": 1_200_000,
        "charged_input_tokens": 1_200_000,
        "actual_output_tokens": 30_000,
        "charged_output_tokens": 30_000,
    }
    if any(
        not isinstance(value := budget.get(field), int)
        or isinstance(value, bool)
        or not 0 <= value <= limit
        for field, limit in token_limits.items()
    ):
        raise ValueError("candidate budget이 승인된 token 상한을 벗어났습니다")
    numeric_limits = {
        "actual_cost_usd": 0.01,
        "charged_cost_usd": 0.01,
        "wall_seconds": 7_200,
    }
    if any(
        not isinstance(value := budget.get(field), (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= limit
        for field, limit in numeric_limits.items()
    ):
        raise ValueError("candidate budget이 승인된 비용·시간 상한을 벗어났습니다")
    for index, call in enumerate(calls, start=1):
        if (
            not isinstance(call, dict)
            or call.get("requested_model") != requested_model
            or call.get("expected_actual_model") != actual_model
            or call.get("actual_model") != actual_model
            or call.get("actual_model_matches_expected") is not True
            or call.get("provider_status") != "provider_response_received"
            or call.get("request_number") != index
            or call.get("attempt_number") != index
            or call.get("retry_count") != 0
            or call.get("error_type") is not None
            or call.get("error_message") is not None
            or call.get("budget_violations") != []
            or not isinstance(call.get("raw_response"), dict)
            or not isinstance(call.get("response_received_at"), str)
            or not call["response_received_at"]
            or any(
                not isinstance(value := call.get(field), int)
                or isinstance(value, bool)
                or value < 0
                for field in ("input_tokens", "output_tokens")
            )
            or any(
                not isinstance(value := call.get(field), (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                for field in ("actual_cost_usd", "latency_ms")
            )
        ):
            raise ValueError(f"candidate call {index}의 model·상태·번호·telemetry가 다릅니다")
    totals = {
        "actual_input_tokens": sum(call["input_tokens"] for call in calls),
        "actual_output_tokens": sum(call["output_tokens"] for call in calls),
        "actual_cost_usd": sum(call["actual_cost_usd"] for call in calls),
    }
    for field, total in totals.items():
        charged_field = f"charged_{field.removeprefix('actual_')}"
        if not math.isclose(
            budget[field], total, rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(
            budget[charged_field], total, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("candidate call telemetry 합계와 summary budget이 다릅니다")
    if (
        sum(call["latency_ms"] for call in calls)
        > budget["wall_seconds"] * 1000 + 1e-6
    ):
        raise ValueError("candidate call latency 합계와 summary budget이 다릅니다")


def load_complete_candidate_run(
    run_directory: str | Path,
    project_root: str | Path,
) -> tuple[list[JudgePair], dict, dict[str, Path], dict[str, str]]:
    """30쌍·60호출 candidate run의 파일과 provenance를 fail-closed 검증한다."""

    run_dir = Path(run_directory).resolve()
    root = Path(project_root).resolve()
    paths = {
        "candidate_summary": run_dir / "candidate-summary.json",
        "candidate_calls": run_dir / "candidate-calls.jsonl",
        "candidate_results": run_dir / "candidate-results.jsonl",
        "baseline_prompt_snapshot": run_dir / "open-cqa-answer-baseline.md",
        "improved_prompt_snapshot": run_dir / "open-cqa-answer-improved.md",
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("candidate run 필수 파일이 없습니다: " + ", ".join(missing))
    try:
        summary = json.loads(paths["candidate_summary"].read_text(encoding="utf-8"))
        calls = _jsonl(paths["candidate_calls"])
        pairs = load_candidate_pairs(paths["candidate_results"])
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise ValueError(f"candidate run을 읽을 수 없습니다: {exc}") from exc
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected_model = "nvidia_nim/google/gemma-4-31b-it"
    expected_actual_model = "google/gemma-4-31b-it"
    required = {
        "artifact_schema_version": 2,
        "observed_status": "complete",
        "probe_only": False,
        "evidence_kind": "live_quality",
        "pair_count": 30,
        "completed_pair_count": 30,
        "expected_request_count": 60,
        "actual_request_count": 60,
        "maximum_request_count": 60,
        "actual_attempt_count": 60,
        "maximum_attempt_count": 60,
        "max_retries_per_request": 0,
        "git_clean": True,
        "requested_model": expected_model,
        "expected_actual_model": expected_actual_model,
        "actual_models": [expected_actual_model],
        "reference_sent_to_task_model": False,
        "candidate_call_record_count": 60,
    }
    changed = [field for field, expected in required.items() if summary.get(field) != expected]
    invalid_output_count = sum(
        status == "invalid_output"
        for pair in pairs
        for status in (
            pair.candidate_a_validation_status,
            pair.candidate_b_validation_status,
        )
    )
    expected_status = "fail" if invalid_output_count else "pass"
    if summary.get("status") != expected_status:
        changed.append("status")
    if summary.get("invalid_output_count") != invalid_output_count:
        changed.append("invalid_output_count")
    if changed:
        raise ValueError("candidate run이 완전한 30쌍 live 실행이 아닙니다: " + ", ".join(changed))
    if len(pairs) != 30 or len(calls) != 60:
        raise ValueError("candidate run에는 30쌍과 60개 call 기록이 필요합니다")
    _validate_candidate_execution(
        summary,
        calls,
        requested_model=expected_model,
        actual_model=expected_actual_model,
    )
    pair_ids = [pair.pair_id for pair in pairs]
    if summary.get("pair_ids") != pair_ids or summary.get("pair_numbers") != list(range(1, 31)):
        raise ValueError("candidate summary의 pair ID·순서가 candidate-results와 다릅니다")
    if summary.get("source_split_counts") != dict(Counter(pair.course_split for pair in pairs)):
        raise ValueError("candidate summary의 split 개수가 candidate-results와 다릅니다")
    candidate_hash = candidate_set_sha256(pairs)
    hash_bindings = {
        "candidate_set_sha256": candidate_hash,
        "candidate_results_sha256": hashes["candidate_results"],
        "candidate_calls_sha256": hashes["candidate_calls"],
        "baseline_prompt_sha256": hashes["baseline_prompt_snapshot"],
        "baseline_prompt_snapshot_sha256": hashes["baseline_prompt_snapshot"],
        "improved_prompt_sha256": hashes["improved_prompt_snapshot"],
        "improved_prompt_snapshot_sha256": hashes["improved_prompt_snapshot"],
    }
    changed_hashes = [
        field for field, expected in hash_bindings.items() if summary.get(field) != expected
    ]
    if changed_hashes:
        raise ValueError(
            "candidate summary hash가 현재 파일과 다릅니다: "
            + ", ".join(changed_hashes)
        )
    current_bindings = {
        "selection_sha256": root / "data/opencqa/week-03-selection.yaml",
        "cases_sha256": root / "local-data/opencqa/week-03-cases.jsonl",
        "provider_config_sha256": root / "configs/week-03-candidates.yaml",
        "lockfile_sha256": root / "uv.lock",
    }
    for field, path in current_bindings.items():
        if not path.is_file() or summary.get(field) != _sha256(path):
            raise ValueError(f"candidate run의 {field}가 현재 파일과 다릅니다")
    components = summary.get("workflow_component_sha256")
    if not isinstance(components, dict) or not components:
        raise ValueError("candidate run의 workflow component hash가 없습니다")
    for relative, digest in components.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"candidate workflow component가 현재 파일과 다릅니다: {relative}")
    call_ids = [str(call.get("sample_id")) for call in calls]
    expected_call_ids = [
        f"{pair.pair_id}/{source}"
        for pair in pairs
        for source in ("baseline", "improved")
    ]
    if call_ids != expected_call_ids:
        raise ValueError("candidate call ID·순서가 30쌍 baseline/improved와 다릅니다")
    call_by_id = {str(call["sample_id"]): call for call in calls}
    task_hashes = summary.get("task_input_sha256")
    for pair in pairs:
        provenance = (pair.candidate_a_provenance, pair.candidate_b_provenance)
        candidate_records = {
            pair.candidate_a_source: (
                pair.candidate_a,
                pair.candidate_a_output,
                pair.candidate_a_validation_status,
                pair.candidate_a_validation_error,
            ),
            pair.candidate_b_source: (
                pair.candidate_b,
                pair.candidate_b_output,
                pair.candidate_b_validation_status,
                pair.candidate_b_validation_error,
            ),
        }
        if {item.call_id for item in provenance} != {
            f"{pair.pair_id}/baseline",
            f"{pair.pair_id}/improved",
        }:
            raise ValueError(f"{pair.pair_id}: candidate call provenance가 다릅니다")
        if any(
            item.requested_model != expected_model
            or item.expected_actual_model != expected_actual_model
            or item.actual_model != expected_actual_model
            or item.prompt_sha256 != summary.get(f"{item.source}_prompt_sha256")
            or item.input_sha256 != (task_hashes or {}).get(pair.pair_id)
            for item in provenance
        ):
            raise ValueError(f"{pair.pair_id}: candidate model·prompt·input provenance가 다릅니다")
        for item in provenance:
            content = _candidate_response_content(call_by_id[item.call_id])
            expected = candidate_records[item.source]
            actual = validate_candidate_output(content)
            if actual != expected:
                raise ValueError(f"{pair.pair_id}: candidate 원문·검증 상태·hash가 다릅니다")
    return pairs, summary, paths, hashes


def load_judge_trials(path: str | Path) -> list[JudgeTrial]:
    trials = [JudgeTrial.model_validate(item) for item in _jsonl(path)]
    if len({(item.pair_id, item.trial) for item in trials}) != len(trials):
        raise ValueError("같은 pair_id와 trial이 중복되었습니다")
    return trials


def load_individual_human_label(path: str | Path) -> IndividualHumanLabel:
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"사람 사전 label 파일이 없습니다: {source}")
    try:
        return IndividualHumanLabel.model_validate(
            yaml.safe_load(source.read_text(encoding="utf-8"))
        )
    except (ValidationError, yaml.YAMLError) as exc:
        raise ValueError(f"사람 사전 label을 확인하세요: {exc}") from exc


def validate_individual_human_label(
    label: IndividualHumanLabel,
    pairs: list[JudgePair],
    expected_candidate_set_sha256: str | None = None,
) -> JudgePair:
    """사람 라벨을 전체 후보 세트와 배정 번호에 묶는다."""

    expected_hash = expected_candidate_set_sha256 or candidate_set_sha256(pairs)
    if label.candidate_set_sha256 != expected_hash:
        raise ValueError("사람 사전 label의 candidate_set_sha256이 현재 후보 세트와 다릅니다")
    if label.pair_number > len(pairs) or pairs[label.pair_number - 1].pair_id != label.pair_id:
        raise ValueError("사람 사전 label의 pair_number·pair_id가 배정 pair와 다릅니다")
    return pairs[label.pair_number - 1]


def _winner_source(
    pair: JudgePair,
    judge_label: Preference | Literal["review"],
) -> ResolvedWinner:
    sources = {pair.candidate_a_source, pair.candidate_b_source}
    if sources != {"baseline", "improved"}:
        raise ValueError(f"{pair.pair_id}: baseline과 improved 후보가 하나씩 필요합니다")
    if judge_label == "candidate_a":
        return pair.candidate_a_source
    if judge_label == "candidate_b":
        return pair.candidate_b_source
    return judge_label


def compare(
    pairs: list[JudgePair],
    judge_trials: list[JudgeTrial],
    *,
    human_label: IndividualHumanLabel | None = None,
    candidate_set_hash: str | None = None,
    live_quality: bool = False,
) -> ComparisonSummary:
    pair_ids = {pair.pair_id for pair in pairs}
    trials = {(item.pair_id, item.trial): item for item in judge_trials}
    if not pairs or len(pair_ids) != len(pairs):
        raise ValueError("pair는 비어 있거나 pair_id가 중복될 수 없습니다")
    if {pair_id for pair_id, _trial in trials} != pair_ids or any(
        (pair_id, trial) not in trials for pair_id in pair_ids for trial in (1, 2)
    ):
        raise ValueError("모든 pair에 Judge trial 1과 2가 필요합니다")
    if human_label is not None:
        expected_hash = candidate_set_hash or candidate_set_sha256(pairs)
        if human_label.candidate_set_sha256 != expected_hash:
            raise ValueError("사람 사전 label의 candidate_set_sha256이 현재 후보 세트와 다릅니다")
        if human_label.pair_id not in pair_ids:
            raise ValueError("사람 사전 label의 pair_id가 비교 대상에 없습니다")

    audits: list[PairAudit] = []
    for pair in pairs:
        first, second = trials[(pair.pair_id, 1)], trials[(pair.pair_id, 2)]
        first_label = first.winner_ab if first.winner_ab == first.winner_ba else "review"
        second_label = second.winner_ab if second.winner_ab == second.winner_ba else "review"
        order_conflict = "review" in (first_label, second_label)
        repetition_conflict = (
            first.winner_ab != second.winner_ab or first.winner_ba != second.winner_ba
        )
        judge_label = first_label if not order_conflict and not repetition_conflict else "review"
        assigned_human = (
            human_label if human_label and human_label.pair_id == pair.pair_id else None
        )
        audits.append(
            PairAudit(
                pair_id=pair.pair_id,
                candidate_a_source=pair.candidate_a_source,
                candidate_b_source=pair.candidate_b_source,
                judge_label=judge_label,
                winner=_winner_source(pair, judge_label),
                order_conflict=order_conflict,
                repetition_conflict=repetition_conflict,
                individual_human_label=assigned_human.label if assigned_human else None,
                judge_agrees_with_individual_human=(
                    None
                    if assigned_human is None or judge_label == "review"
                    else judge_label == assigned_human.label
                ),
            )
        )

    counts = {winner: sum(item.winner == winner for item in audits) for winner in (
        "baseline", "improved", "tie", "review"
    )}
    order_conflicts = sum(item.order_conflict for item in audits)
    repetition_conflicts = sum(item.repetition_conflict for item in audits)
    human_audit = next(
        (item for item in audits if item.individual_human_label is not None),
        None,
    )
    reasons: list[str] = []
    if len(pairs) < 30:
        reasons.append(f"pair_count={len(pairs)} < 30")
    if order_conflicts:
        reasons.append(f"A/B-B/A conflict={order_conflicts}")
    if repetition_conflicts:
        reasons.append(f"repeat conflict={repetition_conflicts}")
    if not live_quality:
        reasons.append("30쌍 live_quality 실행 증거가 아님")
    reasons.append(
        "개인 사람 라벨 한 건은 수업 비교용이며 Judge 보정을 뜻하지 않음"
        if human_audit
        else "개인 사람 라벨이 연결되지 않아 사람 일치 여부를 계산하지 않음"
    )
    reasons.append("독립된 30쌍 사람 보정과 risk label이 없어 blocking에 사용하지 않음")
    return ComparisonSummary(
        pair_count=len(pairs),
        judge_execution_evidence_kind="live_quality" if live_quality else "exploratory",
        baseline_wins=counts["baseline"],
        improved_wins=counts["improved"],
        ties=counts["tie"],
        reviews=counts["review"],
        order_conflicts=order_conflicts,
        repetition_conflicts=repetition_conflicts,
        individual_human_label_count=int(human_audit is not None),
        individual_human_agreement=(
            None
            if human_audit is None or human_audit.judge_agrees_with_individual_human is None
            else float(human_audit.judge_agrees_with_individual_human)
        ),
        human_calibrated=False,
        blocking_eligible=False,
        recommended_use="classroom_demo",
        reasons=reasons,
        pairs=audits,
    )
