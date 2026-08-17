import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from verifiable_ai_workflow.judge_comparison import (
    IndividualHumanLabel,
    JudgeTrial,
    compare,
    load_complete_candidate_run,
    validate_individual_human_label,
)
from verifiable_ai_workflow.open_cqa_candidates import (
    CandidatePairDraft,
    CandidateProvenance,
    bind_candidate_set_sha256,
    validate_candidate_output,
)
from verifiable_ai_workflow.schemas import Evidence, StructuredAnswer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output(answer: str) -> StructuredAnswer:
    return StructuredAnswer(
        answer=answer,
        evidence=[Evidence(evidence_id="chart-1", quote="42%", page_number=1)],
        confidence=0.9,
    )


def _provenance(pair_id: str, source: str, prompt_hash: str, input_hash: str):
    return CandidateProvenance(
        source=source,
        call_id=f"{pair_id}/{source}",
        requested_model="nvidia_nim/google/gemma-4-31b-it",
        expected_actual_model="google/gemma-4-31b-it",
        actual_model="google/gemma-4-31b-it",
        prompt_file=f"open-cqa-answer-{source}.md",
        prompt_sha256=prompt_hash,
        input_sha256=input_hash,
    )


def _pairs(count: int = 30):
    baseline_hash, improved_hash = "a" * 64, "b" * 64
    drafts = []
    for index in range(count):
        pair_id = f"pair-{index + 1:02d}"
        input_hash = f"{index + 1:064x}"
        a_source, b_source = (
            ("baseline", "improved") if index % 2 == 0 else ("improved", "baseline")
        )
        output_a, output_b = _output("A"), _output("B")
        hashes = {"baseline": baseline_hash, "improved": improved_hash}
        drafts.append(
            CandidatePairDraft(
                pair_id=pair_id,
                sample_id=str(index + 1),
                family_id=f"family-{index + 1:02d}",
                course_split=(
                    "development" if index < 18 else "validation" if index < 24 else "test"
                ),
                source_split="val",
                source_revision="a" * 40,
                source_license="GPL-3.0",
                image_sha256=f"{index + 1:064x}",
                image_path=f"local-data/opencqa/images/{index + 1}.jpg",
                question="question",
                reference_answer="reference",
                candidate_a="A",
                candidate_b="B",
                candidate_a_source=a_source,
                candidate_b_source=b_source,
                candidate_a_output=output_a.model_dump(mode="json"),
                candidate_b_output=output_b.model_dump(mode="json"),
                candidate_a_validation_status="valid_output",
                candidate_b_validation_status="valid_output",
                candidate_a_provenance=_provenance(
                    pair_id, a_source, hashes[a_source], input_hash
                ),
                candidate_b_provenance=_provenance(
                    pair_id, b_source, hashes[b_source], input_hash
                ),
            )
        )
    return bind_candidate_set_sha256(drafts)


def _trials(pairs) -> list[JudgeTrial]:
    return [
        JudgeTrial(
            pair_id=pair.pair_id,
            trial=trial,
            winner_ab="candidate_a",
            reason_ab="첫 후보가 더 정확함",
            winner_ba="candidate_a",
            reason_ba="첫 후보가 더 정확함",
        )
        for pair in pairs
        for trial in (1, 2)
    ]


def _candidate_run(tmp_path: Path, *, invalid_output: bool = False):
    root = tmp_path / "project"
    run = root / "reports/week-03/student-full/learner-time/candidates"
    run.mkdir(parents=True)
    current = {
        "selection_sha256": root / "data/opencqa/week-03-selection.yaml",
        "cases_sha256": root / "local-data/opencqa/week-03-cases.jsonl",
        "provider_config_sha256": root / "configs/week-03-candidates.yaml",
        "lockfile_sha256": root / "uv.lock",
    }
    for index, path in enumerate(current.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"current-{index}\n", encoding="utf-8")
    component = root / "scripts/run_open_cqa_candidates.py"
    component.parent.mkdir(parents=True)
    component.write_text("component\n", encoding="utf-8")
    baseline = run / "open-cqa-answer-baseline.md"
    improved = run / "open-cqa-answer-improved.md"
    baseline.write_text("baseline prompt", encoding="utf-8")
    improved.write_text("improved prompt", encoding="utf-8")
    baseline_hash, improved_hash = _sha(baseline), _sha(improved)
    pairs = _pairs()
    pairs = [
        pair.model_copy(
            update={
                "candidate_a_provenance": pair.candidate_a_provenance.model_copy(
                    update={
                        "prompt_sha256": (
                            baseline_hash
                            if pair.candidate_a_source == "baseline"
                            else improved_hash
                        )
                    }
                ),
                "candidate_b_provenance": pair.candidate_b_provenance.model_copy(
                    update={
                        "prompt_sha256": (
                            baseline_hash
                            if pair.candidate_b_source == "baseline"
                            else improved_hash
                        )
                    }
                ),
            }
        )
        for pair in pairs
    ]
    raw_overrides: dict[tuple[str, str], str] = {}
    if invalid_output:
        invalid_raw = json.dumps(
            {
                "answer": "The chart does not answer this question.",
                "evidence": [],
                "confidence": 0.0,
                "abstained": True,
                "abstention_reason": "차트에서 답을 확인할 수 없음",
                "tool_requests": [],
            }
        )
        candidate, output, status, error = validate_candidate_output(invalid_raw)
        first = pairs[0]
        side = "a" if first.candidate_a_source == "baseline" else "b"
        raw_overrides[(first.pair_id, "baseline")] = invalid_raw
        pairs[0] = first.model_copy(
            update={
                f"candidate_{side}": candidate,
                f"candidate_{side}_output": output,
                f"candidate_{side}_validation_status": status,
                f"candidate_{side}_validation_error": error,
            }
        )
    pairs = bind_candidate_set_sha256(
        [
            CandidatePairDraft.model_validate(
                pair.model_dump(exclude={"candidate_set_sha256"})
            )
            for pair in pairs
        ]
    )
    results = run / "candidate-results.jsonl"
    results.write_text("".join(pair.model_dump_json() + "\n" for pair in pairs), encoding="utf-8")
    calls = run / "candidate-calls.jsonl"

    def response_content(pair, source: str) -> str:
        if override := raw_overrides.get((pair.pair_id, source)):
            return override
        output = (
            pair.candidate_a_output
            if pair.candidate_a_source == source
            else pair.candidate_b_output
        )
        return json.dumps(output, ensure_ascii=False, separators=(",", ":"))

    calls.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"{pair.pair_id}/{source}",
                    "requested_model": "nvidia_nim/google/gemma-4-31b-it",
                    "expected_actual_model": "google/gemma-4-31b-it",
                    "actual_model": "google/gemma-4-31b-it",
                    "actual_model_matches_expected": True,
                    "provider_status": "provider_response_received",
                    "latency_ms": 100.0,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "actual_cost_usd": 0.0,
                    "retry_count": 0,
                    "request_number": index,
                    "attempt_number": index,
                    "response_received_at": "2026-08-17T00:00:00+00:00",
                    "error_type": None,
                    "error_message": None,
                    "budget_violations": [],
                    "raw_response": {"content": response_content(pair, source)},
                }
            )
            + "\n"
            for index, (pair, source) in enumerate(
                (
                    (pair, source)
                    for pair in pairs
                    for source in ("baseline", "improved")
                ),
                start=1,
            )
        ),
        encoding="utf-8",
    )
    budget = {
        "request_count": 60,
        "attempt_count": 60,
        "reserved_input_tokens": 1_200_000,
        "reserved_output_tokens": 30_000,
        "reserved_cost_usd": 0.0,
        "actual_input_tokens": 600,
        "actual_output_tokens": 300,
        "actual_cost_usd": 0.0,
        "charged_input_tokens": 600,
        "charged_output_tokens": 300,
        "charged_cost_usd": 0.0,
        "wall_seconds": 60.0,
    }
    summary = {
        "artifact_schema_version": 2,
        "status": "fail" if invalid_output else "pass",
        "observed_status": "complete",
        "probe_only": False,
        "evidence_kind": "live_quality",
        "pair_count": 30,
        "pair_numbers": list(range(1, 31)),
        "pair_ids": [pair.pair_id for pair in pairs],
        "completed_pair_count": 30,
        "source_split_counts": dict(Counter(pair.course_split for pair in pairs)),
        "expected_request_count": 60,
        "actual_request_count": 60,
        "maximum_request_count": 60,
        "actual_attempt_count": 60,
        "maximum_attempt_count": 60,
        "max_retries_per_request": 0,
        "git_sha": "1" * 40,
        "git_clean": True,
        "requested_model": "nvidia_nim/google/gemma-4-31b-it",
        "expected_actual_model": "google/gemma-4-31b-it",
        "actual_models": ["google/gemma-4-31b-it"],
        "reference_sent_to_task_model": False,
        "candidate_call_record_count": 60,
        "invalid_output_count": int(invalid_output),
        "budget": budget,
        "candidate_set_sha256": pairs[0].candidate_set_sha256,
        "candidate_results_sha256": _sha(results),
        "candidate_calls_sha256": _sha(calls),
        "baseline_prompt_sha256": baseline_hash,
        "baseline_prompt_snapshot_sha256": baseline_hash,
        "improved_prompt_sha256": improved_hash,
        "improved_prompt_snapshot_sha256": improved_hash,
        "task_input_sha256": {
            pair.pair_id: pair.candidate_a_provenance.input_sha256 for pair in pairs
        },
        "workflow_component_sha256": {
            "scripts/run_open_cqa_candidates.py": _sha(component)
        },
        **{field: _sha(path) for field, path in current.items()},
    }
    (run / "candidate-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root, run, pairs


def test_comparison_counts_baseline_improved_ties_and_no_codex_reference() -> None:
    pairs = _pairs()
    summary = compare(pairs, _trials(pairs), live_quality=True)

    assert summary.baseline_wins == 15
    assert summary.improved_wins == 15
    assert summary.ties == summary.reviews == 0
    assert summary.human_calibrated is False
    assert summary.blocking_eligible is False
    assert summary.recommended_use == "classroom_demo"
    assert "codex" not in summary.model_dump_json().casefold()


def test_order_or_repeat_change_goes_to_review() -> None:
    pairs = _pairs()
    trials = _trials(pairs)
    trials[0] = trials[0].model_copy(update={"winner_ba": "candidate_b"})

    summary = compare(pairs, trials)

    assert summary.pairs[0].judge_label == "review"
    assert summary.pairs[0].winner == "review"
    assert summary.order_conflicts == 1
    assert summary.repetition_conflicts == 1


def test_individual_human_agreement_is_bound_to_candidate_set() -> None:
    pairs = _pairs()
    label = IndividualHumanLabel(
        pair_number=2,
        pair_id=pairs[1].pair_id,
        candidate_set_sha256=pairs[0].candidate_set_sha256,
        reviewer_id="learner",
        label="candidate_a",
        reason="차트의 수치와 비교 대상을 직접 확인했습니다.",
    )

    validate_individual_human_label(label, pairs)
    summary = compare(
        pairs,
        _trials(pairs),
        human_label=label,
        candidate_set_hash=pairs[0].candidate_set_sha256,
    )

    assert summary.individual_human_label_count == 1
    assert summary.individual_human_agreement == 1
    assert summary.pairs[1].judge_agrees_with_individual_human is True
    with pytest.raises(ValueError, match="candidate_set_sha256"):
        validate_individual_human_label(
            label.model_copy(update={"candidate_set_sha256": "0" * 64}),
            pairs,
        )


def test_missing_second_trial_fails() -> None:
    pairs = _pairs()
    with pytest.raises(ValueError, match="trial 1과 2"):
        compare(pairs, _trials(pairs)[:-1])


def test_complete_candidate_run_binds_all_files_and_current_inputs(tmp_path: Path) -> None:
    root, run, expected_pairs = _candidate_run(tmp_path)

    pairs, summary, paths, hashes = load_complete_candidate_run(run, root)

    assert pairs == expected_pairs
    assert summary["actual_request_count"] == 60
    assert set(paths) == set(hashes) == {
        "candidate_summary",
        "candidate_calls",
        "candidate_results",
        "baseline_prompt_snapshot",
        "improved_prompt_snapshot",
    }


def test_candidate_run_rejects_raw_response_that_disagrees_with_result(
    tmp_path: Path,
) -> None:
    root, run, _pairs_value = _candidate_run(tmp_path)
    calls_path = run / "candidate-calls.jsonl"
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    output = json.loads(calls[0]["raw_response"]["content"])
    output["answer"] = "changed answer"
    calls[0]["raw_response"]["content"] = json.dumps(output)
    calls_path.write_text("".join(json.dumps(call) + "\n" for call in calls))
    summary_path = run / "candidate-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["candidate_calls_sha256"] = _sha(calls_path)
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="candidate 원문"):
        load_complete_candidate_run(run, root)


def test_candidate_run_rejects_budget_over_cap(tmp_path: Path) -> None:
    root, run, _pairs_value = _candidate_run(tmp_path)
    summary_path = run / "candidate-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["budget"]["actual_input_tokens"] = 1_200_001
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="budget"):
        load_complete_candidate_run(run, root)


def test_candidate_run_rejects_changed_call_model(tmp_path: Path) -> None:
    root, run, _pairs_value = _candidate_run(tmp_path)
    calls_path = run / "candidate-calls.jsonl"
    calls = [
        json.loads(line)
        for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    calls[0]["actual_model"] = "another-model"
    calls_path.write_text(
        "".join(json.dumps(call) + "\n" for call in calls),
        encoding="utf-8",
    )
    summary_path = run / "candidate-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["candidate_calls_sha256"] = _sha(calls_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="model·상태·번호·telemetry"):
        load_complete_candidate_run(run, root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "inconclusive", "완전한 30쌍"),
        ("actual_request_count", 59, "완전한 30쌍"),
        ("git_clean", False, "완전한 30쌍"),
        ("reference_sent_to_task_model", True, "완전한 30쌍"),
    ],
)
def test_candidate_run_rejects_partial_or_untrusted_summary(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    root, run, _pairs_value = _candidate_run(tmp_path)
    summary_path = run / "candidate-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_complete_candidate_run(run, root)


def test_candidate_run_accepts_complete_quality_failure(tmp_path: Path) -> None:
    root, run, expected_pairs = _candidate_run(tmp_path, invalid_output=True)

    pairs, summary, _paths, _hashes = load_complete_candidate_run(run, root)

    assert pairs == expected_pairs
    assert summary["status"] == "fail"
    assert summary["invalid_output_count"] == 1


def test_candidate_run_rejects_replaced_prompt_snapshot(tmp_path: Path) -> None:
    root, run, _pairs_value = _candidate_run(tmp_path)
    (run / "open-cqa-answer-baseline.md").write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        load_complete_candidate_run(run, root)
