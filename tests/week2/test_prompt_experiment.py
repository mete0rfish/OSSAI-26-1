from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verifiable_ai_workflow.prompt_comparison import compare_prompt_runs
from verifiable_ai_workflow.schemas import EvaluationResult

MODEL = "google/gemma-4-31b-it"
REQUESTED_MODEL = "nvidia_nim/google/gemma-4-31b-it"
CONTROLLED_HASHES = {
    "git_sha": "a" * 40,
    "dataset_sha256": "b" * 64,
    "input_manifest_content_sha256": "c" * 64,
    "lockfile_sha256": "d" * 64,
    "schema_sha256": "e" * 64,
    "scorer_sha256": "f" * 64,
    "workflow_sha256": "1" * 64,
}


def _result(index: int, *, success: bool) -> EvaluationResult:
    scores = {
        "task_success": float(success),
        "answer_correct": float(success),
        "schema_validity": 1.0,
        "json_object_only": 1.0,
        "numeric_match": 1.0,
        "quote_answer_support": 1.0,
    }
    return EvaluationResult(
        sample_id=f"sample-{index:02d}",
        family_id="family-01",
        status="passed" if success else "failed",
        scores=scores,
        reasons={},
        evidence_kind="live_quality",
        provider_status="success",
        model_call={"actual_model": MODEL},
    )


def _write_run(
    root: Path,
    *,
    run_id: str,
    prompt_hash: str,
    first_success: bool,
    git_sha: str = "a" * 40,
    pricing_verified_on: str = "2026-08-15",
) -> None:
    root.mkdir()
    prompt_path = root / "prompt.md"
    prompt_path.write_text(prompt_hash + "\n", encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    results = [_result(index, success=(first_success or index > 0)) for index in range(40)]
    (root / "results.jsonl").write_text(
        "".join(result.model_dump_json() + "\n" for result in results),
        encoding="utf-8",
    )
    budget = {
        "request_count": 40,
        "attempt_count": 40,
        "reserved_input_tokens": 400000,
        "reserved_output_tokens": 20000,
        "reserved_cost_usd": 0.0,
        "actual_input_tokens": 200000,
        "actual_output_tokens": 10000,
        "actual_cost_usd": 0.0,
        "charged_input_tokens": 400000,
        "charged_output_tokens": 20000,
        "charged_cost_usd": 0.0,
        "wall_seconds": 60.0,
    }
    summary = {
        "run_id": run_id,
        "status": "pass" if first_success else "fail",
        "observed_status": "complete",
        "probe_only": False,
        "record_count": 40,
        "target_count": 40,
        "evidence_kind": "live_quality",
        "evaluation_mode": "benchmark",
        "fallback_enabled": False,
        "replay_enabled": False,
        "live_call_performed": True,
        "requested_model": REQUESTED_MODEL,
        "expected_actual_model": MODEL,
        "actual_models": [MODEL],
        "provider_error_count": 0,
        "model_drift_count": 0,
        "budget": budget,
        "provenance": {
            **CONTROLLED_HASHES,
            "git_sha": git_sha,
            "prompt_sha256": prompt_hash,
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "status": "pass" if first_success else "fail",
        "observed_status": "complete",
        "input_manifest_sha256": "4" * 64,
        "budget": budget,
        "contract": {
            "run_id": run_id,
            "provider": {
                "adapter": "litellm",
                "requested_model": REQUESTED_MODEL,
                "expected_actual_model": MODEL,
                "structured_output": "prompt_only",
                "pricing_verified_on": pricing_verified_on,
            },
            "evaluation_mode": "benchmark",
            "evidence_kind": "live_quality",
            "fallback_enabled": False,
            "replay_enabled": False,
            "caps": {
                "max_requests": 40,
                "max_attempts": 40,
                "max_input_tokens": 800000,
                "max_output_tokens": 20000,
                "max_cost_usd": 0.01,
                "max_wall_seconds": 7200.0,
            },
            "provenance": summary["provenance"],
        },
    }
    (root / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )


def test_gemma_prompt_has_no_fenced_example_and_requires_short_answer(
    project_root: Path,
) -> None:
    prompt = (project_root / "prompts/pdf-question-answer-gemma4.md").read_text(encoding="utf-8")

    assert "```" not in prompt
    assert "값과 단위만" in prompt
    assert "연도·분기·항목을 반복하지 말고" in prompt
    assert "두 번째 JSON을 절대 출력하지 않습니다" in prompt
    assert '"answer":"답변 보류"' in prompt


def test_prompt_comparison_accepts_only_prompt_treatment(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, run_id="baseline-run", prompt_hash="2" * 64, first_success=False)
    _write_run(
        candidate,
        run_id="candidate-run",
        prompt_hash="3" * 64,
        first_success=True,
        pricing_verified_on="2026-08-16",
    )

    report = compare_prompt_runs(baseline, candidate)

    assert report.invalid_reasons == []
    assert report.automated_status == "pass"
    assert report.score_source == "stored_results"
    assert report.effective_scorer_sha256 is None
    assert report.metric_deltas["task_success"].delta_percentage_points == 2.5
    assert report.classification_counts == {
        "new_success": 1,
        "new_failure": 0,
        "unchanged": 39,
        "not_comparable": 0,
    }
    assert report.new_success_ids == ["sample-00"]
    assert report.new_failure_ids == []


def test_prompt_comparison_rejects_git_change(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, run_id="baseline-run", prompt_hash="2" * 64, first_success=False)
    _write_run(
        candidate,
        run_id="candidate-run",
        prompt_hash="3" * 64,
        first_success=True,
        git_sha="9" * 40,
    )

    report = compare_prompt_runs(baseline, candidate)

    assert report.automated_status == "inconclusive"
    assert "prompt 외 통제값 불일치: git_sha" in report.invalid_reasons


def test_prompt_comparison_rejects_scorer_change(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, run_id="baseline-run", prompt_hash="2" * 64, first_success=False)
    _write_run(candidate, run_id="candidate-run", prompt_hash="3" * 64, first_success=True)
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["provenance"]["scorer_sha256"] = "9" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = compare_prompt_runs(baseline, candidate)

    assert report.automated_status == "inconclusive"
    assert "prompt 외 통제값 불일치: scorer_sha256" in report.invalid_reasons


def test_prompt_comparison_rejects_summary_manifest_provenance_mismatch(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, run_id="baseline-run", prompt_hash="2" * 64, first_success=False)
    _write_run(candidate, run_id="candidate-run", prompt_hash="3" * 64, first_success=True)
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["provenance"]["git_sha"] = "9" * 40
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = compare_prompt_runs(baseline, candidate)

    assert report.automated_status == "inconclusive"
    assert (
        "candidate summary와 run manifest의 provenance가 다릅니다"
        in report.invalid_reasons
    )


def test_prompt_comparison_rejects_changed_prompt_snapshot(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, run_id="baseline-run", prompt_hash="2" * 64, first_success=False)
    _write_run(candidate, run_id="candidate-run", prompt_hash="3" * 64, first_success=True)
    (candidate / "prompt.md").write_text("실행 뒤 바뀜\n", encoding="utf-8")

    report = compare_prompt_runs(baseline, candidate)

    assert report.automated_status == "inconclusive"
    assert "candidate 저장 prompt.md와 provenance hash가 다릅니다" in report.invalid_reasons


def test_prompt_comparison_rejects_budget_over_cap(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, run_id="baseline-run", prompt_hash="2" * 64, first_success=False)
    _write_run(candidate, run_id="candidate-run", prompt_hash="3" * 64, first_success=True)
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["budget"]["actual_input_tokens"] = 800001
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = compare_prompt_runs(baseline, candidate)

    assert report.automated_status == "inconclusive"
    assert "candidate budget actual_input_tokens=800001, 상한=800000" in report.invalid_reasons


def test_prompt_comparison_excludes_provider_error_from_quality_denominator(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, run_id="baseline-run", prompt_hash="2" * 64, first_success=False)
    _write_run(candidate, run_id="candidate-run", prompt_hash="3" * 64, first_success=True)

    results_path = candidate / "results.jsonl"
    results = [
        EvaluationResult.model_validate_json(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    results[0] = results[0].model_copy(
        update={
            "status": "inconclusive",
            "provider_status": "provider_error",
            "scores": {name: 0.0 for name in results[0].scores},
            "model_call": {"actual_model": None},
        }
    )
    results_path.write_text(
        "".join(result.model_dump_json() + "\n" for result in results),
        encoding="utf-8",
    )
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "status": "inconclusive",
            "observed_status": "inconclusive",
            "provider_error_count": 1,
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = candidate / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "inconclusive"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_prompt_runs(baseline, candidate)

    assert report.automated_status == "inconclusive"
    assert report.quality_eligible_counts == {"baseline": 40, "candidate": 39}
    assert report.provider_error_counts == {"baseline": 0, "candidate": 1}
    assert report.classification_counts == {
        "new_success": 0,
        "new_failure": 0,
        "unchanged": 39,
        "not_comparable": 1,
    }
    assert report.not_comparable_ids == ["sample-00"]
    assert report.metric_deltas["task_success"].candidate == 1.0
