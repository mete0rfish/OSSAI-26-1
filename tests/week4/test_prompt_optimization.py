import argparse
import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepeval.models import DeepEvalBaseLLM
from deepeval.prompt import Prompt

from scripts import (
    inspect_week_04_prompt_results,
    optimize_open_cqa_prompt,
    prepare_week_04_lab,
)
from verifiable_ai_workflow.config.settings import load_settings
from verifiable_ai_workflow.open_cqa_candidates import OpenCQACase
from verifiable_ai_workflow.prompt_optimization import (
    OpenCqaDeterministicMetric,
    OpenCqaVlmCallback,
    build_prompt_optimizer,
    score_output,
    split_goldens,
    validate_development_goldens,
)
from verifiable_ai_workflow.week4_materials import _project_path


class NoCallModel(DeepEvalBaseLLM):
    def load_model(self):
        return self

    def get_model_name(self, *args, **kwargs):
        return "no-call"

    def generate(self, *args, **kwargs):
        raise AssertionError("factory에서 model을 호출하면 안 됩니다")

    async def a_generate(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


def _cases() -> list[OpenCQACase]:
    return [
        OpenCQACase(
            pair_id=f"pair-{index}",
            sample_id=str(index),
            family_id=f"family-{index}",
            course_split=("development" if index < 18 else "validation" if index < 24 else "test"),
            source_split="val",
            source_revision="a" * 40,
            source_license="GPL-3.0",
            image_sha256="b" * 64,
            image_path=f"{index}.png",
            question="What changed?",
            reference_answer="It rose from 10% to 20%.",
        )
        for index in range(30)
    ]


def test_split_is_18_6_6_and_optimizer_uses_development(project_root: Path) -> None:
    splits = split_goldens(list(reversed(_cases())))

    assert {name: len(items) for name, items in splits.items()} == {
        "development": 18,
        "validation": 6,
        "test": 6,
    }
    assert all(
        (golden.additional_metadata or {})["split"] == split
        for split, goldens in splits.items()
        for golden in goldens
    )
    assert all(
        (golden.additional_metadata or {})["image_sha256"] == "b" * 64
        for goldens in splits.values()
        for golden in goldens
    )
    optimizer = build_prompt_optimizer(
        goldens=splits["development"],
        model_callback=lambda prompt, golden: "{}",
        optimizer_model=NoCallModel(),
        config_path=project_root / "configs/week-04.yaml",
    )
    assert optimizer.algorithm.iterations == 2
    demo_optimizer = build_prompt_optimizer(
        goldens=splits["development"][:2],
        model_callback=lambda prompt, golden: "{}",
        optimizer_model=NoCallModel(),
        config_path=project_root / "configs/week-04-demo.yaml",
    )
    assert demo_optimizer.algorithm.iterations == 1
    assert demo_optimizer.algorithm.minibatch_size == 1
    with pytest.raises(ValueError, match="development"):
        validate_development_goldens(splits["validation"])


def test_metric_returns_feedback_for_missing_number() -> None:
    golden = split_goldens(_cases())["development"][0]
    output = json.dumps(
        {
            "answer": "It rose to 20%.",
            "evidence": [{"evidence_id": "chart#page=1", "quote": "20%", "page_number": 1}],
            "confidence": 0.8,
            "abstained": False,
            "abstention_reason": None,
            "tool_requests": [],
        }
    )

    result = score_output(OpenCqaDeterministicMetric(), golden, output)

    assert result["pair_id"] == "pair-0"
    assert result["sample_id"] == "0"
    assert result["image_path"] == "0.png"
    assert result["split"] == "development"
    assert 0 < result["score"] < 1
    assert "10%" in result["reason"]


def test_baseline_prompt_interpolates_question(project_root: Path) -> None:
    prompt = Prompt(text_template=(project_root / "prompts/week-04-baseline.md").read_text())
    rendered = prompt.interpolate(question="What changed?")
    assert "What changed?" in rendered
    assert all(field in rendered for field in ("evidence", "abstained", "답변 보류"))


def test_vlm_callback_labels_jpeg_input_correctly(tmp_path: Path) -> None:
    image = tmp_path / "chart.jpg"
    image.write_bytes(b"jpeg")

    class Provider:
        messages: list[dict] | None = None

        def generate(self, sample_id, messages):
            assert sample_id == "0"
            self.messages = messages
            return "{}"

    provider = Provider()
    golden = split_goldens(_cases())["development"][0]
    golden.additional_metadata["image_path"] = image.name
    golden.additional_metadata["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()

    OpenCqaVlmCallback(provider, tmp_path)(Prompt(text_template="{question}"), golden)

    assert provider.messages is not None
    data_url = provider.messages[1]["content"][1]["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")

    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        OpenCqaVlmCallback(provider, tmp_path)(Prompt(text_template="{question}"), golden)


def test_optimizer_separates_nim_target_and_gemini_review() -> None:
    target = load_settings(optimize_open_cqa_prompt.TARGET_CONFIG)
    optimizer = load_settings(optimize_open_cqa_prompt.OPTIMIZER_CONFIG)

    assert target.provider.model == "nvidia_nim/google/gemma-4-31b-it"
    assert optimizer.provider.model == "gemini/gemini-3.5-flash-lite"


def test_identical_candidate_cannot_win_from_repeated_model_variation() -> None:
    baseline = Prompt(text_template="same {question}")
    candidate = Prompt(text_template="same {question}")

    selected, prompt, reason = optimize_open_cqa_prompt._select_prompt(
        baseline, candidate, baseline_mean=0.1, candidate_mean=0.9
    )

    assert (selected, prompt, reason) == (
        "baseline",
        baseline,
        "candidate_identical",
    )


def test_week_04_student_inputs_reject_path_traversal() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="별칭"):
        prepare_week_04_lab._student_alias("../other")
    with pytest.raises(ValueError, match="상대 경로"):
        _project_path(Path("/project"), "../other")


def test_week_04_prepare_rejects_changed_artifact_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("original", encoding="utf-8")
    stored = {"artifact.json": hashlib.sha256(artifact.read_bytes()).hexdigest()}

    assert prepare_week_04_lab._stored_hashes_match(stored, {"artifact.json": artifact})
    artifact.write_text("changed", encoding="utf-8")
    assert not prepare_week_04_lab._stored_hashes_match(stored, {"artifact.json": artifact})


def test_optimizer_detects_changed_source_file(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("original", encoding="utf-8")
    expected = [(source, hashlib.sha256(source.read_bytes()).hexdigest())]

    assert not optimize_open_cqa_prompt._files_changed(expected)
    source.write_text("changed", encoding="utf-8")
    assert optimize_open_cqa_prompt._files_changed(expected)


def test_demo_output_path_is_generated_automatically() -> None:
    output = optimize_open_cqa_prompt._output_path(None, 2)

    assert output.parent == optimize_open_cqa_prompt.PROJECT_ROOT / "reports/week-04"
    datetime.strptime(output.name.removeprefix("class-demo-"), "%Y%m%d-%H%M%S")
    with pytest.raises(ValueError, match="전체 평가"):
        optimize_open_cqa_prompt._output_path(None, None)


def test_student_preparation_hides_full_result_until_after_demo(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        prepare_week_04_lab,
        "prepare",
        lambda *args, **kwargs: {
            "materials_label": "verified",
            "source_git_sha": "abcdef0",
            "student_root": Path("local-data/week-04-students/minsu"),
            "report_root": Path("reports/week-04/students/minsu"),
        },
    )
    monkeypatch.setattr(sys, "argv", ["prepare_week_04_lab.py", "--alias", "minsu"])

    assert prepare_week_04_lab.main() == 0
    output = capsys.readouterr().out
    assert "개발 사례 2건 시연 후 전체 저장 결과 확인" in output
    assert all(
        label not in output
        for label in ("선택 결과", "실제 API 호출", "지시문 비교 자료", "이미지 응답 자료")
    )


def test_week_04_inspector_finds_prompt_and_score_changes() -> None:
    assert inspect_week_04_prompt_results._changed_lines(
        "answer: 값\nkeep", "answer: 문장\nkeep"
    ) == ["-answer: 값", "+answer: 문장"]
    comparisons = inspect_week_04_prompt_results._comparisons(
        [
            {"sample_id": "1", "prompt": "baseline", "score": 0.2},
            {"sample_id": "1", "prompt": "candidate", "score": 0.7},
        ]
    )
    assert comparisons[0]["delta"] == pytest.approx(0.5)
    best, worst = inspect_week_04_prompt_results._representatives(
        [
            {"sample_id": "up", "delta": 0.2},
            {"sample_id": "down", "delta": -0.3},
            {"sample_id": "same", "delta": 0.0},
        ]
    )
    assert (best["sample_id"], worst["sample_id"]) == ("up", "down")


def test_week_04_inspector_handles_identical_candidate(monkeypatch, tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    (tmp_path / "prompts").mkdir()
    (tmp_path / "local-data/opencqa").mkdir(parents=True)
    result_dir.mkdir()
    baseline = "same {question}\n"
    (tmp_path / "prompts/week-04-baseline.md").write_text(baseline, encoding="utf-8")
    (tmp_path / "local-data/opencqa/week-03-cases.jsonl").write_text(
        "placeholder\n", encoding="utf-8"
    )
    (result_dir / "candidate-prompt.md").write_text(baseline, encoding="utf-8")
    (result_dir / "selected-prompt.md").write_text(baseline, encoding="utf-8")
    (result_dir / "calls.jsonl").write_text(
        '{"provider_role":"target"}\n{"provider_role":"optimizer"}\n',
        encoding="utf-8",
    )
    (result_dir / "validation.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": str(index),
                    "prompt": "baseline",
                    "score": 0.5,
                    "output": "{}",
                }
            )
            + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    summary = {
        "selected": "baseline",
        "git_sha": "a" * 40,
        "observed_status": "complete",
        "development_count": 18,
        "validation_count": 6,
        "test_count": 6,
        "test_used_for_generation_or_selection": False,
        "target_provider": {
            "requested_model": "nim",
            "actual_models": ["nim"],
        },
        "optimizer_provider": {
            "requested_model": "gemini",
            "actual_models": ["gemini"],
        },
        "provider_error_count": 0,
        "model_drift_count": 0,
        "selection_reason": "candidate_identical",
        "baseline_mean": 0.5,
        "candidate_mean": None,
        "candidate_changed": False,
        "artifact_sha256": {
            name: inspect_week_04_prompt_results._sha256(path)
            for name, path in {
                "calls.jsonl": result_dir / "calls.jsonl",
                "validation.jsonl": result_dir / "validation.jsonl",
                "candidate-prompt.md": result_dir / "candidate-prompt.md",
                "selected-prompt.md": result_dir / "selected-prompt.md",
                "week-03-cases.jsonl": (tmp_path / "local-data/opencqa/week-03-cases.jsonl"),
            }.items()
        },
    }
    (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(
        inspect_week_04_prompt_results,
        "load_week4_class_materials",
        lambda project_root: SimpleNamespace(label="test", prompt_optimization_dir=result_dir),
    )
    monkeypatch.setattr(inspect_week_04_prompt_results, "load_open_cqa_cases", lambda path: [])

    output = inspect_week_04_prompt_results.inspect(tmp_path, Path("result"))

    assert "새 지시문 평균: 지시문이 같아 실행하지 않음" in output
    assert "지시문이 같아 새 답을 만들지 않았습니다." in output

    (result_dir / "validation.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256"):
        inspect_week_04_prompt_results.inspect(tmp_path, Path("result"))


@pytest.mark.parametrize(
    ("catalog_date", "pricing_date"),
    [
        ("2000-01-01", date.today().isoformat()),
        (date.today().isoformat(), "2000-01-01"),
    ],
)
def test_optimizer_rejects_stale_preflight_before_live_work(
    monkeypatch,
    tmp_path,
    catalog_date,
    pricing_date,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_open_cqa_prompt.py",
            "--live-optimize",
            "--max-requests",
            "9",
            "--max-input-tokens",
            "180000",
            "--max-output-tokens",
            "4500",
            "--max-cost-usd",
            "0.01",
            "--max-wall-seconds",
            "600",
            "--catalog-verified-on",
            catalog_date,
            "--pricing-verified-on",
            pricing_date,
            "--optimizer-max-requests",
            "4",
            "--optimizer-max-attempts",
            "8",
            "--optimizer-max-input-tokens",
            "40000",
            "--optimizer-max-output-tokens",
            "16000",
            "--optimizer-max-cost-usd",
            "0.01",
            "--optimizer-max-wall-seconds",
            "7200",
            "--optimizer-catalog-verified-on",
            date.today().isoformat(),
            "--optimizer-pricing-verified-on",
            date.today().isoformat(),
            "--output",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="실행 당일"):
        optimize_open_cqa_prompt.main()


def test_optimizer_rejects_larger_than_approved_caps(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_open_cqa_prompt.py",
            "--live-optimize",
            "--max-requests",
            "46",
            "--max-input-tokens",
            "900000",
            "--max-output-tokens",
            "22500",
            "--max-cost-usd",
            "0.01",
            "--max-wall-seconds",
            "7200",
            "--catalog-verified-on",
            date.today().isoformat(),
            "--pricing-verified-on",
            date.today().isoformat(),
            "--optimizer-max-requests",
            "4",
            "--optimizer-max-attempts",
            "8",
            "--optimizer-max-input-tokens",
            "40000",
            "--optimizer-max-output-tokens",
            "16000",
            "--optimizer-max-cost-usd",
            "0.01",
            "--optimizer-max-wall-seconds",
            "7200",
            "--optimizer-catalog-verified-on",
            date.today().isoformat(),
            "--optimizer-pricing-verified-on",
            date.today().isoformat(),
            "--output",
            str(tmp_path / "oversized"),
        ],
    )

    with pytest.raises(SystemExit, match="승인 cap"):
        optimize_open_cqa_prompt.main()


@pytest.mark.parametrize(
    ("attempt_count", "expected_status", "provider_error"),
    [
        (1, "partial", "APIConnectionError"),
        (0, "not_run", "LiveBudgetExceeded"),
    ],
)
def test_optimizer_failure_without_response_is_inconclusive(
    monkeypatch, tmp_path, attempt_count, expected_status, provider_error
) -> None:
    class Provider:
        structured_output = "json_schema"

        def __init__(self, settings, output_ceiling) -> None:
            self.model = settings.provider.model
            self.expected_actual_model = settings.provider.expected_actual_model
            self.request_output_token_ceiling = output_ceiling
            self.last_call = {
                "provider_status": "provider_error" if attempt_count else "blocked",
                "error_type": provider_error,
            }
            self.budget = SimpleNamespace(
                summary=lambda: {
                    "request_count": attempt_count,
                    "attempt_count": attempt_count,
                }
            )

    class Optimizer:
        def optimize(self, prompt, goldens):
            del prompt, goldens
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(optimize_open_cqa_prompt, "_clean_git", lambda: "a" * 40)
    monkeypatch.setattr(optimize_open_cqa_prompt, "load_project_env", lambda path: path)
    monkeypatch.setattr(
        optimize_open_cqa_prompt,
        "build_course_provider",
        lambda settings, caps, **kwargs: Provider(
            settings,
            kwargs.get("request_output_token_ceiling")
            or settings.limits.request_output_token_ceiling,
        ),
    )
    monkeypatch.setattr(
        optimize_open_cqa_prompt,
        "load_open_cqa_cases",
        lambda path: _cases(),
    )
    monkeypatch.setattr(
        optimize_open_cqa_prompt,
        "build_prompt_optimizer",
        lambda **kwargs: Optimizer(),
    )
    output = tmp_path / "failed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_open_cqa_prompt.py",
            "--live-optimize",
            "--max-requests",
            "45",
            "--max-input-tokens",
            "900000",
            "--max-output-tokens",
            "22500",
            "--max-cost-usd",
            "0.01",
            "--max-wall-seconds",
            "7200",
            "--catalog-verified-on",
            date.today().isoformat(),
            "--pricing-verified-on",
            date.today().isoformat(),
            "--optimizer-max-requests",
            "4",
            "--optimizer-max-attempts",
            "8",
            "--optimizer-max-input-tokens",
            "40000",
            "--optimizer-max-output-tokens",
            "16000",
            "--optimizer-max-cost-usd",
            "0.01",
            "--optimizer-max-wall-seconds",
            "7200",
            "--optimizer-catalog-verified-on",
            date.today().isoformat(),
            "--optimizer-pricing-verified-on",
            date.today().isoformat(),
            "--output",
            str(output),
        ],
    )

    assert optimize_open_cqa_prompt.main() == 2
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "inconclusive"
    assert summary["observed_status"] == expected_status
    assert summary["run_mode"] == "full_evaluation"
    assert summary["error_type"] == "RuntimeError"
    assert summary["source_revision"] == "a" * 40
    assert len(summary["split_sample_ids"]["test"]) == 6
    assert summary["target_provider"]["pricing_verified_on"] == date.today().isoformat()
    assert summary["optimizer_provider"]["pricing_verified_on"] == date.today().isoformat()
    assert summary["target_provider"]["requested_model"].startswith("nvidia_nim/")
    assert summary["optimizer_provider"]["requested_model"].startswith("gemini/")
    assert summary["artifact_sha256"]["calls.jsonl"]
    assert summary["baseline_prompt_sha256"]
    assert summary["schema_sha256"]
    assert summary["scorer_sha256"]
    assert summary["optimizer_config_sha256"]
    calls = [
        json.loads(line)
        for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {call["provider_role"] for call in calls} == {"target", "optimizer"}
    assert all(call["error_type"] == provider_error for call in calls)
