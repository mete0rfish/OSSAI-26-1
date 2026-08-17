import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from scripts import compare_open_cqa_judge, inspect_judge_pair, run_open_cqa_judge
from scripts.run_open_cqa_judge import maximum_requests
from verifiable_ai_workflow.config.settings import load_settings
from verifiable_ai_workflow.judge_comparison import IndividualHumanLabel, JudgeTrial
from verifiable_ai_workflow.open_cqa_candidates import (
    CandidatePairDraft,
    CandidateProvenance,
    bind_candidate_set_sha256,
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


def _pairs():
    drafts = []
    for index in range(30):
        pair_id = f"pair-{index + 1:02d}"
        input_hash = f"{index + 1:064x}"
        a_source, b_source = (
            ("baseline", "improved") if index % 2 == 0 else ("improved", "baseline")
        )

        def provenance(
            source: str,
            pair_id: str = pair_id,
            input_hash: str = input_hash,
        ):
            return CandidateProvenance(
                source=source,
                call_id=f"{pair_id}/{source}",
                requested_model="nvidia_nim/google/gemma-4-31b-it",
                expected_actual_model="google/gemma-4-31b-it",
                actual_model="google/gemma-4-31b-it",
                prompt_file=f"open-cqa-answer-{source}.md",
                prompt_sha256=("a" if source == "baseline" else "b") * 64,
                input_sha256=input_hash,
            )

        output_a, output_b = _output("후보 A 답"), _output("후보 B 답")
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
                question="질문",
                reference_answer="숨겨야 하는 기준 답",
                candidate_a="후보 A 답",
                candidate_b="후보 B 답",
                candidate_a_source=a_source,
                candidate_b_source=b_source,
                candidate_a_output=output_a.model_dump(mode="json"),
                candidate_b_output=output_b.model_dump(mode="json"),
                candidate_a_validation_status="valid_output",
                candidate_b_validation_status="valid_output",
                candidate_a_provenance=provenance(a_source),
                candidate_b_provenance=provenance(b_source),
            )
        )
    return bind_candidate_set_sha256(drafts)


def _human_label_file(
    monkeypatch,
    tmp_path: Path,
    pairs,
    *,
    pair_number: int = 17,
) -> Path:
    root = tmp_path / "local-data/week-03-student-judges"
    path = root / "learner/human-label.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        IndividualHumanLabel(
            pair_number=pair_number,
            pair_id=pairs[pair_number - 1].pair_id,
            candidate_set_sha256=pairs[0].candidate_set_sha256,
            reviewer_id="learner",
            label="candidate_a",
            reason="차트의 수치와 비교 대상을 직접 확인했습니다.",
        ).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_open_cqa_judge, "STUDENT_LABEL_ROOT", root.resolve())
    return path


def test_two_trials_and_two_orders_allow_at_most_eight_requests_per_pair() -> None:
    assert maximum_requests(1) == 8
    assert maximum_requests(30) == 240


def test_pair_selection_requires_full_candidate_set() -> None:
    pairs = _pairs()

    assert run_open_cqa_judge._select_pairs(pairs, 1, 17) == [pairs[16]]
    assert run_open_cqa_judge._select_pairs(pairs, 30, 1) == pairs
    with pytest.raises(SystemExit, match="정확히 30쌍"):
        run_open_cqa_judge._select_pairs(pairs[:-1], 1, 1)


def test_human_label_is_locked_to_candidate_set_and_assigned_pair(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pairs = _pairs()
    path = _human_label_file(monkeypatch, tmp_path, pairs)

    resolved, label, pair = run_open_cqa_judge._load_locked_human_label(
        path,
        pairs,
        pairs[0].candidate_set_sha256,
    )

    assert resolved == path
    assert pair == pairs[16]
    assert label.label == "candidate_a"
    with pytest.raises(SystemExit, match="candidate_set_sha256"):
        run_open_cqa_judge._load_locked_human_label(path, pairs, "0" * 64)


def test_blind_inspector_hides_reference_source_and_model(monkeypatch, capsys) -> None:
    pairs = _pairs()
    monkeypatch.setattr(inspect_judge_pair, "load_candidate_pairs", lambda path: pairs)
    monkeypatch.setattr(
        sys,
        "argv",
        ["inspect_judge_pair.py", "--candidates", "candidate-results.jsonl", "--number", "1"],
    )

    assert inspect_judge_pair.main() == 0
    output = capsys.readouterr().out
    assert "[평가표 ID] pair-01" in output
    assert "후보 A 답" in output and "후보 B 답" in output
    assert pairs[0].candidate_set_sha256 in output
    assert "숨겨야 하는 기준 답" not in output
    assert "baseline" not in output and "improved" not in output
    assert "gemma" not in output.casefold()


def test_inspector_reveals_reference_and_sources_only_after_locked_label(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pairs = _pairs()
    label = _human_label_file(monkeypatch, tmp_path, pairs)
    monkeypatch.setattr(inspect_judge_pair, "load_candidate_pairs", lambda path: pairs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_judge_pair.py",
            "--candidates",
            "candidate-results.jsonl",
            "--number",
            "17",
            "--human-label",
            str(label),
            "--reveal",
        ],
    )

    assert inspect_judge_pair.main() == 0
    output = capsys.readouterr().out
    assert "[기대 답] 숨겨야 하는 기준 답" in output
    assert "[후보 A 출처] baseline" in output
    assert "[후보 B 출처] improved" in output
    assert "[학습자 선택]" not in output


def test_judge_config_uses_free_tier_flash_lite_and_exact_caps(
    monkeypatch,
    project_root: Path,
) -> None:
    settings = load_settings(
        project_root / "configs/google-gemini-3.5-flash-lite-judge.yaml"
    )

    assert settings.provider.model == "gemini/gemini-3.5-flash-lite"
    assert settings.provider.expected_actual_model == "gemini-3.5-flash-lite"
    assert settings.provider.api_base == "https://generativelanguage.googleapis.com/v1beta"
    assert settings.provider.api_key_env == "GEMINI_API_KEY"
    assert settings.provider.sampling_parameters == "omit"
    assert settings.provider.billing_basis == "free_tier"
    assert settings.provider.input_cost_per_token_usd == 0.0
    assert settings.provider.output_cost_per_token_usd == 0.0
    assert settings.limits.requests_per_minute == 15
    assert settings.limits.max_cost_usd == 0.01
    assert settings.limits.request_input_token_ceiling == 5_000
    assert settings.limits.request_output_token_ceiling == 500
    loaded, _provider_config, _rubric, structured_output = (
        run_open_cqa_judge._load_approved_settings()
    )
    assert loaded == settings
    assert structured_output == "json_schema"

    changed = settings.model_copy(
        update={"limits": settings.limits.model_copy(update={"requests_per_minute": 14})}
    )
    monkeypatch.setattr(run_open_cqa_judge, "load_settings", lambda path: changed)
    with pytest.raises(SystemExit, match="cap"):
        run_open_cqa_judge._load_approved_settings()


def test_judge_rejects_stale_preflight_before_candidate_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_open_cqa_judge.py",
            "--live-judge",
            "--candidate-run",
            str(tmp_path / "candidates"),
            "--pair-limit",
            "30",
            "--max-requests",
            "240",
            "--max-retries",
            "1",
            "--max-input-tokens",
            "1200000",
            "--max-output-tokens",
            "120000",
            "--max-cost-usd",
            "0.01",
            "--max-wall-seconds",
            "10800",
            "--catalog-verified-on",
            "2000-01-01",
            "--pricing-verified-on",
            date.today().isoformat(),
            "--output",
            str(tmp_path / "judge"),
        ],
    )

    with pytest.raises(SystemExit, match="실행 당일"):
        run_open_cqa_judge.main()


def test_judge_rejects_larger_than_approved_caps(monkeypatch, tmp_path) -> None:
    pairs = _pairs()
    monkeypatch.setattr(
        run_open_cqa_judge,
        "load_complete_candidate_run",
        lambda run, root: (
            pairs,
            {"git_sha": "1" * 40, "status": "pass", "invalid_output_count": 0},
            {},
            {"candidate_results": "a" * 64},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_open_cqa_judge.py",
            "--live-judge",
            "--candidate-run",
            str(tmp_path / "candidates"),
            "--pair-limit",
            "30",
            "--max-requests",
            "241",
            "--max-retries",
            "1",
            "--max-input-tokens",
            "1200000",
            "--max-output-tokens",
            "120000",
            "--max-cost-usd",
            "0.01",
            "--max-wall-seconds",
            "10800",
            "--catalog-verified-on",
            date.today().isoformat(),
            "--pricing-verified-on",
            date.today().isoformat(),
            "--output",
            str(tmp_path / "judge"),
        ],
    )

    with pytest.raises(SystemExit, match="승인 cap"):
        run_open_cqa_judge.main()


class _Budget:
    def __init__(self) -> None:
        self.requests = 0

    def summary(self):
        return {"request_count": self.requests, "attempt_count": self.requests}


class _JudgeProvider:
    captured_kwargs = []

    def __init__(self, **kwargs) -> None:
        self.captured_kwargs.append(kwargs)
        self.model = kwargs["model"]
        self.expected_actual_model = kwargs["expected_actual_model"]
        self.callback = kwargs["on_response_received"]
        self.budget = _Budget()
        self.last_call = None

    def emit_tie(self) -> None:
        self.budget.requests += 1
        self.last_call = {"provider_status": "success"}
        self.callback(dict(self.last_call))


@pytest.mark.parametrize(
    ("pair_limit", "pair_number", "caps", "expected_status", "expected_requests"),
    [
        (1, 17, (8, 40_000, 4_000, 0.01, 300), "inconclusive", 4),
    ],
)
def test_runner_accepts_assigned_human_label_and_passes_sampling_omit(
    monkeypatch,
    tmp_path: Path,
    pair_limit: int,
    pair_number: int,
    caps,
    expected_status: str,
    expected_requests: int,
) -> None:
    pairs = _pairs()
    human_label = _human_label_file(monkeypatch, tmp_path, pairs)
    evidence_files = {}
    evidence_hashes = {}
    for name in (
        "candidate_summary",
        "candidate_calls",
        "candidate_results",
        "baseline_prompt_snapshot",
        "improved_prompt_snapshot",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        evidence_files[name] = path
        evidence_hashes[name] = _sha(path)
    monkeypatch.setattr(
        run_open_cqa_judge,
        "load_complete_candidate_run",
        lambda run, root: (
            pairs,
            {"git_sha": "1" * 40, "status": "fail", "invalid_output_count": 1},
            evidence_files,
            evidence_hashes,
        ),
    )
    monkeypatch.setattr(run_open_cqa_judge, "_git_state", lambda: ("1" * 40, False))
    monkeypatch.setattr(
        run_open_cqa_judge,
        "_validate_pair_images",
        lambda selected, **kwargs: {
            pair.pair_id: tmp_path / f"{pair.pair_id}.jpg" for pair in selected
        },
    )
    monkeypatch.setattr(run_open_cqa_judge, "_changed_inputs", lambda expected: [])
    monkeypatch.setattr(run_open_cqa_judge, "load_project_env", lambda root: root)
    monkeypatch.setattr(run_open_cqa_judge, "LiteLLMProvider", _JudgeProvider)

    def fake_measure(metric, pair, **kwargs):
        del pair, kwargs
        metric.model.provider.emit_tie()
        return "tie", "두 응답이 같습니다"

    monkeypatch.setattr(run_open_cqa_judge, "measure", fake_measure)
    max_requests, max_input, max_output, max_cost, max_wall = caps
    output = tmp_path / f"judge-{pair_limit}"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_open_cqa_judge.py",
            "--live-judge",
            "--candidate-run",
            str(tmp_path / "candidate-run"),
            "--pair-limit",
            str(pair_limit),
            "--pair-number",
            str(pair_number),
            "--human-label",
            str(human_label),
            "--max-requests",
            str(max_requests),
            "--max-retries",
            "1",
            "--max-input-tokens",
            str(max_input),
            "--max-output-tokens",
            str(max_output),
            "--max-cost-usd",
            str(max_cost),
            "--max-wall-seconds",
            str(max_wall),
            "--catalog-verified-on",
            date.today().isoformat(),
            "--pricing-verified-on",
            date.today().isoformat(),
            "--output",
            str(output),
        ],
    )

    assert run_open_cqa_judge.main() == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == expected_status
    assert summary["observed_status"] == "complete"
    assert summary["actual_request_count"] == expected_requests
    assert summary["candidate_run_validated_complete"] is True
    assert summary["candidate_git_sha"] == "1" * 40
    assert summary["candidate_status"] == "fail"
    assert summary["candidate_invalid_output_count"] == 1
    assert summary["candidate_summary_sha256"] == evidence_hashes["candidate_summary"]
    assert summary["human_label_pair_number"] == 17
    assert summary["sampling_parameters"] == "omit"
    assert _JudgeProvider.captured_kwargs[-1]["sampling_parameters"] == "omit"
    assert _JudgeProvider.captured_kwargs[-1]["max_attempts"] == max_requests


def test_full_judge_rejects_human_label_until_compare(monkeypatch, tmp_path: Path) -> None:
    pairs = _pairs()
    human_label = _human_label_file(monkeypatch, tmp_path, pairs)
    monkeypatch.setattr(
        run_open_cqa_judge,
        "load_complete_candidate_run",
        lambda run, root: (
            pairs,
            {"git_sha": "1" * 40, "status": "pass", "invalid_output_count": 0},
            {},
            {"candidate_results": "a" * 64},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_open_cqa_judge.py",
            "--live-judge",
            "--candidate-run",
            str(tmp_path / "candidate-run"),
            "--pair-limit",
            "30",
            "--human-label",
            str(human_label),
            "--max-requests",
            "240",
            "--max-retries",
            "1",
            "--max-input-tokens",
            "1200000",
            "--max-output-tokens",
            "120000",
            "--max-cost-usd",
            "0.01",
            "--max-wall-seconds",
            "10800",
            "--catalog-verified-on",
            date.today().isoformat(),
            "--pricing-verified-on",
            date.today().isoformat(),
            "--output",
            str(tmp_path / "judge"),
        ],
    )

    with pytest.raises(SystemExit, match="compare 단계"):
        run_open_cqa_judge.main()


def _judge_trials(pairs) -> list[JudgeTrial]:
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


def test_full_comparison_verifies_candidate_evidence_and_one_human_label(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pairs = _pairs()
    candidate_run = tmp_path / "reports/week-03/student-full/learner-time/candidates"
    candidate_run.mkdir(parents=True)
    candidate_hashes = {
        "candidate_summary": "1" * 64,
        "candidate_calls": "2" * 64,
        "candidate_results": "3" * 64,
        "baseline_prompt_snapshot": "4" * 64,
        "improved_prompt_snapshot": "5" * 64,
    }
    monkeypatch.setattr(
        compare_open_cqa_judge,
        "load_complete_candidate_run",
        lambda run, root: (
            pairs,
            {"git_sha": "1" * 40, "status": "fail", "invalid_output_count": 1},
            {},
            candidate_hashes,
        ),
    )
    monkeypatch.setattr(compare_open_cqa_judge, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(compare_open_cqa_judge, "CONFIG", tmp_path / "configs/week-03-judge.yaml")
    monkeypatch.setattr(
        compare_open_cqa_judge,
        "STUDENT_LABEL_ROOT",
        (tmp_path / "local-data/week-03-student-judges").resolve(),
    )
    config = tmp_path / "configs/week-03-judge.yaml"
    provider_config = tmp_path / compare_open_cqa_judge.APPROVED_PROVIDER_CONFIG
    rubric = tmp_path / compare_open_cqa_judge.APPROVED_RUBRIC
    for path, value in (
        (config, "judge config"),
        (provider_config, "provider config"),
        (rubric, "rubric"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    results = tmp_path / "judge/judge-results.jsonl"
    results.parent.mkdir()
    results.write_text(
        "".join(trial.model_dump_json() + "\n" for trial in _judge_trials(pairs)),
        encoding="utf-8",
    )
    calls = results.with_name("judge-calls.jsonl")
    sample_ids = [
        f"{pair.pair_id}/trial-{trial}/{order}"
        for pair in pairs
        for trial in (1, 2)
        for order in ("ab", "ba")
    ]
    call_records = [
        {
            "sample_id": sample_id,
            "provider_status": "provider_response_received",
            "actual_model": compare_open_cqa_judge.EXPECTED_ACTUAL_MODEL,
            "actual_model_matches_expected": True,
            "raw_response": {"id": f"response-{request_number}"},
            "response_id": f"response-{request_number}",
            "response_received_at": "2026-08-17T00:00:00+00:00",
            "request_number": request_number,
            "attempt_number": request_number,
            "input_tokens": 1,
            "output_tokens": 1,
            "actual_cost_usd": 0.0,
            "latency_ms": 1.0,
        }
        for request_number, sample_id in enumerate(sample_ids, start=1)
    ]
    calls.write_text(
        "".join(json.dumps(call) + "\n" for call in call_records),
        encoding="utf-8",
    )
    label_root = tmp_path / "local-data/week-03-student-judges"
    label = label_root / "learner/human-label.yaml"
    label.parent.mkdir(parents=True)
    label.write_text(
        IndividualHumanLabel(
            pair_number=17,
            pair_id=pairs[16].pair_id,
            candidate_set_sha256=pairs[0].candidate_set_sha256,
            reviewer_id="learner",
            label="candidate_a",
            reason="차트의 수치와 비교 대상을 직접 확인했습니다.",
        ).model_dump_json(),
        encoding="utf-8",
    )
    label_sha256 = _sha(label)
    summary = {
        "status": "pass",
        "observed_status": "complete",
        "probe_only": False,
        "evidence_kind": "live_quality",
        "pair_count": 30,
        "completed_pair_count": 30,
        "completed_trial_count": 60,
        "expected_request_count": 120,
        "actual_request_count": 120,
        "maximum_request_count": 240,
        "actual_attempt_count": 120,
        "maximum_attempt_count": 240,
        "max_retries_per_request": 1,
        "pair_ids": [pair.pair_id for pair in pairs],
        "git_sha": "1" * 40,
        "git_dirty": False,
        "model": compare_open_cqa_judge.APPROVED_MODEL,
        "expected_actual_model": compare_open_cqa_judge.EXPECTED_ACTUAL_MODEL,
        "sampling_parameters": "omit",
        "billing_basis": "free_tier",
        "input_cost_per_token_usd": 0.0,
        "output_cost_per_token_usd": 0.0,
        "reference_answer_role": "arena_expected_output",
        "candidate_run_directory": str(candidate_run.resolve()),
        "candidate_run_validated_complete": True,
        "candidate_git_sha": "1" * 40,
        "candidate_status": "fail",
        "candidate_invalid_output_count": 1,
        "candidate_summary_sha256": candidate_hashes["candidate_summary"],
        "candidate_calls_sha256": candidate_hashes["candidate_calls"],
        "candidate_results_sha256": candidate_hashes["candidate_results"],
        "candidate_set_sha256": pairs[0].candidate_set_sha256,
        "candidate_baseline_prompt_snapshot_sha256": candidate_hashes[
            "baseline_prompt_snapshot"
        ],
        "candidate_improved_prompt_snapshot_sha256": candidate_hashes[
            "improved_prompt_snapshot"
        ],
        "judge_results_sha256": _sha(results),
        "judge_calls_sha256": _sha(calls),
        "config_sha256": _sha(config),
        "provider_config_sha256": _sha(provider_config),
        "rubric_sha256": _sha(rubric),
        "rubric_path": compare_open_cqa_judge.APPROVED_RUBRIC,
        "budget": {
            "request_count": 120,
            "attempt_count": 120,
            "reserved_input_tokens": 600_000,
            "actual_input_tokens": 120,
            "charged_input_tokens": 120,
            "reserved_output_tokens": 60_000,
            "actual_output_tokens": 120,
            "charged_output_tokens": 120,
            "reserved_cost_usd": 0.0,
            "actual_cost_usd": 0.0,
            "charged_cost_usd": 0.0,
            "wall_seconds": 60.0,
        },
    }
    results.with_name("summary.json").write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "comparison.json"
    argv = [
        "compare_open_cqa_judge.py",
        "--candidate-run",
        str(candidate_run),
        "--judge-results",
        str(results),
        "--human-label",
        str(label),
        "--human-label-sha256",
        label_sha256,
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert compare_open_cqa_judge.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["judge_execution_evidence_kind"] == "live_quality"
    assert payload["individual_human_label_count"] == 1
    assert payload["candidate_invalid_output_count"] == 1
    assert "task output contract violation=1" in payload["reasons"]
    assert payload["pairs"][16]["judge_agrees_with_individual_human"] is True
    assert payload["blocking_eligible"] is False
    assert "codex" not in json.dumps(payload).casefold()

    monkeypatch.setattr(sys, "argv", [*argv[:5], *argv[9:]])
    with pytest.raises(SystemExit, match="--human-label이 필요합니다"):
        compare_open_cqa_judge.main()

    monkeypatch.setattr(sys, "argv", [*argv[:7], *argv[9:]])
    with pytest.raises(SystemExit, match="--human-label-sha256"):
        compare_open_cqa_judge.main()

    monkeypatch.setattr(sys, "argv", [*argv[:8], "0" * 64, *argv[9:]])
    with pytest.raises(SystemExit, match="잠근 사람 사전 label SHA-256"):
        compare_open_cqa_judge.main()

    calls.write_text(
        "".join(json.dumps(call) + "\n" for call in call_records[:-1]),
        encoding="utf-8",
    )
    summary["judge_calls_sha256"] = _sha(calls)
    results.with_name("summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="live_quality"):
        compare_open_cqa_judge.main()

    call_records[0]["sample_id"] = "wrong/trial-1/ab"
    calls.write_text(
        "".join(json.dumps(call) + "\n" for call in call_records),
        encoding="utf-8",
    )
    summary["judge_calls_sha256"] = _sha(calls)
    results.with_name("summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(SystemExit, match="live_quality"):
        compare_open_cqa_judge.main()
    call_records[0]["sample_id"] = sample_ids[0]

    calls.write_text("{}\n" * 120, encoding="utf-8")
    summary["judge_calls_sha256"] = _sha(calls)
    results.with_name("summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(SystemExit, match="live_quality"):
        compare_open_cqa_judge.main()

    calls.write_text(
        "".join(json.dumps(call) + "\n" for call in call_records),
        encoding="utf-8",
    )
    summary["judge_calls_sha256"] = _sha(calls)
    summary["budget"]["charged_input_tokens"] = 1_200_001
    results.with_name("summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(SystemExit, match="live_quality"):
        compare_open_cqa_judge.main()
